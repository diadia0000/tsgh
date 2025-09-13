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
    
    // Fine-tune all affine parameters with sub-pixel precision
    double bestMI = 0.0;
    Mat bestTransform = result.clone();
    
    RegistrationMetricsCalculator metrics;
    Mat aligned = metrics.applyTransform(mov, result);
    bestMI = metrics.calculateMutualInformation(ref, aligned);
    logProgress("    B-spline initial MI: " + to_string(bestMI));
    
    return performFineTuningOptimization(ref, mov, bestTransform, bestMI);
}

cv::Mat RegistrationStages::performFineTuningOptimization(const cv::Mat& ref, const cv::Mat& mov,
                                                         cv::Mat& bestTransform, double& bestMI) {
    RegistrationMetricsCalculator metrics;
    
    // Multi-level fine optimization with decreasing step sizes
    const double translationSteps[] = {0.5, 0.25, 0.1};
    const double rotationSteps[] = {0.5, 0.25, 0.1};  // degrees
    const double scaleSteps[] = {0.005, 0.002, 0.001};
    
    for (int level = 0; level < 3; level++) {
        logProgress("    B-spline fine-tuning level " + to_string(level + 1) + "/3");
        
        double tStep = translationSteps[level];
        double rStep = rotationSteps[level] * CV_PI / 180.0;
        double sStep = scaleSteps[level];
        
        bool improved = true;
        int iterations = 0;
        const int maxIterations = 15;
        
        while (improved && iterations < maxIterations) {
            improved = false;
            iterations++;
            
            improved = performFineTranslationAdjustment(ref, mov, bestTransform, bestMI, tStep, metrics) || improved;
            improved = performFineRotationAdjustment(ref, mov, bestTransform, bestMI, rStep, metrics) || improved;
            improved = performFineScaleAdjustment(ref, mov, bestTransform, bestMI, sStep, metrics) || improved;
        }
        
        logProgress("    Fine-tuning level " + to_string(level + 1) + " completed, MI: " + to_string(bestMI));
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