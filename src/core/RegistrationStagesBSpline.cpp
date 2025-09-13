#include "wsi/RegistrationStages.h"
#include "wsi/RegistrationMetrics.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>

using namespace std;
using namespace cv;

namespace cell_registration {

cv::Mat RegistrationStages::bsplineAlignment(const cv::Mat& ref, const cv::Mat& mov, const cv::Mat& initial) {
    // Ensure we work with 2x3 matrix
    Mat result;
    if (initial.rows == 2 && initial.cols == 3) {
        result = initial.clone();
    } else {
        result = initial(Rect(0, 0, 3, 2)).clone();
    }
    
    // 改進的灰階預處理 - 針對每張圖片進行像素級優化
    Mat refGray, movGray;
    
    // 轉換為灰階並進行增強預處理
    if (ref.channels() == 3) {
        cvtColor(ref, refGray, COLOR_BGR2GRAY);
    } else {
        refGray = ref.clone();
    }
    
    if (mov.channels() == 3) {
        cvtColor(mov, movGray, COLOR_BGR2GRAY);
    } else {
        movGray = mov.clone();
    }
    
    // 對每張圖片進行像素級增強
    Mat refEnhanced, movEnhanced;
    
    // 使用CLAHE (對比度限制自適應直方圖均衡化)
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(refGray, refEnhanced);
    clahe->apply(movGray, movEnhanced);
    
    // 高斯濾波去噪
    GaussianBlur(refEnhanced, refEnhanced, Size(3, 3), 0.5);
    GaussianBlur(movEnhanced, movEnhanced, Size(3, 3), 0.5);
    
    // Fine-tune all affine parameters with sub-pixel precision
    double bestMI = 0.0;
    Mat bestTransform = result.clone();
    
    RegistrationMetricsCalculator metrics;
    Mat aligned = metrics.applyTransform(movEnhanced, result);
    bestMI = metrics.calculateMutualInformation(refEnhanced, aligned);
    logProgress("    B-spline initial MI (enhanced): " + to_string(bestMI));
    
    // 使用增強後的灰階圖像進行優化
    return performFineTuningOptimization(refEnhanced, movEnhanced, bestTransform, bestMI);
}

cv::Mat RegistrationStages::performFineTuningOptimization(const cv::Mat& ref, const cv::Mat& mov,
                                                         cv::Mat& bestTransform, double& bestMI) {
    RegistrationMetricsCalculator metrics;
    
    // 更精細的多層級優化 - 像素級精度
    const double translationSteps[] = {2.0, 1.0, 0.5, 0.25, 0.1};  // 增加層級
    const double rotationSteps[] = {1.0, 0.5, 0.25, 0.1, 0.05};    // 更精細的角度
    const double scaleSteps[] = {0.01, 0.005, 0.002, 0.001, 0.0005}; // 更精細的縮放
    
    for (int level = 0; level < 5; level++) {  // 增加到5個層級
        logProgress("    B-spline fine-tuning level " + to_string(level + 1) + "/5");
        
        double tStep = translationSteps[level];
        double rStep = rotationSteps[level] * CV_PI / 180.0;
        double sStep = scaleSteps[level];
        
        bool improved = true;
        int iterations = 0;
        const int maxIterations = level < 3 ? 20 : 30;  // 後期層級增加迭代次數
        
        while (improved && iterations < maxIterations) {
            improved = false;
            iterations++;
            
            // 每次迭代都嘗試所有方向的微調
            bool transImproved = performFineTranslationAdjustment(ref, mov, bestTransform, bestMI, tStep, metrics);
            bool rotImproved = performFineRotationAdjustment(ref, mov, bestTransform, bestMI, rStep, metrics);
            bool scaleImproved = performFineScaleAdjustment(ref, mov, bestTransform, bestMI, sStep, metrics);
            
            improved = transImproved || rotImproved || scaleImproved;
            
            // 在精細層級增加組合優化
            if (level >= 3 && improved) {
                // 嘗試小幅度的組合調整
                performCombinedFineAdjustment(ref, mov, bestTransform, bestMI, tStep * 0.5, rStep * 0.5, sStep * 0.5, metrics);
            }
        }
        
        logProgress("    Fine-tuning level " + to_string(level + 1) + " completed, MI: " + to_string(bestMI) + 
                   " (iterations: " + to_string(iterations) + ")");
    }
    
    logFinalBSplineResult(bestTransform, bestMI);
    return bestTransform;
}

bool RegistrationStages::performFineTranslationAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                                         cv::Mat& bestTransform, double& bestMI,
                                                         double tStep, RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // Fine translation adjustments
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            Mat candidate = bestTransform.clone();
            candidate.at<double>(0, 2) += dx * tStep;
            candidate.at<double>(1, 2) += dy * tStep;
            
            Mat testAligned = metrics.applyTransform(mov, candidate);
            double testMI = metrics.calculateMutualInformation(ref, testAligned);
            
            if (testMI > bestMI) {
                bestMI = testMI;
                bestTransform = candidate.clone();
                improved = true;
            }
        }
    }
    
    return improved;
}

bool RegistrationStages::performFineRotationAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                                      cv::Mat& bestTransform, double& bestMI,
                                                      double rStep, RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // Current parameters
    double a = bestTransform.at<double>(0, 0);
    double c = bestTransform.at<double>(1, 0);
    double currentScale = sqrt(a*a + c*c);
    double currentRotation = atan2(c, a);
    
    // Fine rotation adjustments
    for (int dr = -1; dr <= 1; dr += 2) {
        double newRotation = currentRotation + dr * rStep;
        double newCos = cos(newRotation);
        double newSin = sin(newRotation);
        
        Mat candidate = bestTransform.clone();
        candidate.at<double>(0, 0) = currentScale * newCos;
        candidate.at<double>(0, 1) = -currentScale * newSin;
        candidate.at<double>(1, 0) = currentScale * newSin;
        candidate.at<double>(1, 1) = currentScale * newCos;
        
        Mat testAligned = metrics.applyTransform(mov, candidate);
        double testMI = metrics.calculateMutualInformation(ref, testAligned);
        
        if (testMI > bestMI) {
            bestMI = testMI;
            bestTransform = candidate.clone();
            improved = true;
        }
    }
    
    return improved;
}

bool RegistrationStages::performFineScaleAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                                   cv::Mat& bestTransform, double& bestMI,
                                                   double sStep, RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // Current parameters
    double a = bestTransform.at<double>(0, 0);
    double c = bestTransform.at<double>(1, 0);
    double currentScale = sqrt(a*a + c*c);
    double currentRotation = atan2(c, a);
    
    // Fine scale adjustments
    for (int ds = -1; ds <= 1; ds += 2) {
        double newScale = currentScale + ds * sStep;
        if (newScale > 0.7 && newScale < 1.5) {  // Tighter bounds for fine-tuning
            double cos_r = cos(currentRotation);
            double sin_r = sin(currentRotation);
            
            Mat candidate = bestTransform.clone();
            candidate.at<double>(0, 0) = newScale * cos_r;
            candidate.at<double>(0, 1) = -newScale * sin_r;
            candidate.at<double>(1, 0) = newScale * sin_r;
            candidate.at<double>(1, 1) = newScale * cos_r;
            
            Mat testAligned = metrics.applyTransform(mov, candidate);
            double testMI = metrics.calculateMutualInformation(ref, testAligned);
            
            if (testMI > bestMI) {
                bestMI = testMI;
                bestTransform = candidate.clone();
                improved = true;
            }
        }
    }
    
    return improved;
}

bool RegistrationStages::performCombinedFineAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                                      cv::Mat& bestTransform, double& bestMI,
                                                      double tStep, double rStep, double sStep,
                                                      RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // 當前參數
    double a = bestTransform.at<double>(0, 0);
    double c = bestTransform.at<double>(1, 0);
    double currentScale = sqrt(a*a + c*c);
    double currentRotation = atan2(c, a);
    
    // 嘗試小幅度的組合調整
    for (int dt = -1; dt <= 1; dt++) {
        for (int dr = -1; dr <= 1; dr++) {
            for (int ds = -1; ds <= 1; ds++) {
                if (dt == 0 && dr == 0 && ds == 0) continue;
                
                Mat candidate = bestTransform.clone();
                
                // 組合調整
                candidate.at<double>(0, 2) += dt * tStep;
                candidate.at<double>(1, 2) += dt * tStep;
                
                double newRotation = currentRotation + dr * rStep;
                double newScale = currentScale + ds * sStep;
                
                if (newScale > 0.5 && newScale < 2.0) {
                    double cos_r = cos(newRotation);
                    double sin_r = sin(newRotation);
                    
                    candidate.at<double>(0, 0) = newScale * cos_r;
                    candidate.at<double>(0, 1) = -newScale * sin_r;
                    candidate.at<double>(1, 0) = newScale * sin_r;
                    candidate.at<double>(1, 1) = newScale * cos_r;
                    
                    Mat testAligned = metrics.applyTransform(mov, candidate);
                    double testMI = metrics.calculateMutualInformation(ref, testAligned);
                    
                    if (testMI > bestMI) {
                        bestMI = testMI;
                        bestTransform = candidate.clone();
                        improved = true;
                    }
                }
            }
        }
    }
    
    return improved;
}

void RegistrationStages::logFinalBSplineResult(const cv::Mat& bestTransform, double bestMI) {
    // Final summary
    double final_a = bestTransform.at<double>(0, 0);
    double final_c = bestTransform.at<double>(1, 0);
    double final_tx = bestTransform.at<double>(0, 2);
    double final_ty = bestTransform.at<double>(1, 2);
    
    double finalScale = sqrt(final_a*final_a + final_c*final_c);
    double finalRotation = atan2(final_c, final_a) * 180.0 / CV_PI;
    
    logProgress("    B-spline fine-tuning completed:");
    logProgress("      Final MI: " + to_string(bestMI));
    logProgress("      Final scale: " + to_string(finalScale));
    logProgress("      Final rotation: " + to_string(finalRotation) + " degrees");
    logProgress("      Final translation: (" + to_string(final_tx) + ", " + to_string(final_ty) + ")");
}



} // namespace cell_registration