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

cv::Mat RegistrationStages::performMultiParameterOptimization(const cv::Mat& ref, const cv::Mat& mov,
                                                             cv::Mat& bestTransform, double& bestMI) {
    RegistrationMetricsCalculator metrics;
    
    // Multi-parameter optimization: translation, rotation, and scale
    const double translationSteps[] = {2.0, 1.0, 0.5};
    const double rotationSteps[] = {2.0, 1.0, 0.5};  // degrees
    const double scaleSteps[] = {0.02, 0.01, 0.005};
    
    for (int level = 0; level < 3; level++) {
        logProgress("    Optimization level " + to_string(level + 1) + "/3");
        
        double tStep = translationSteps[level];
        double rStep = rotationSteps[level] * CV_PI / 180.0;  // Convert to radians
        double sStep = scaleSteps[level];
        
        bool improved = true;
        int iterations = 0;
        const int maxIterations = 20;
        
        while (improved && iterations < maxIterations) {
            improved = false;
            iterations++;
            
            improved = optimizeTranslation(ref, mov, bestTransform, bestMI, tStep, metrics) || improved;
            improved = optimizeRotation(ref, mov, bestTransform, bestMI, rStep, metrics) || improved;
            improved = optimizeScale(ref, mov, bestTransform, bestMI, sStep, metrics) || improved;
        }
        
        logProgress("    Level " + to_string(level + 1) + " completed, MI: " + to_string(bestMI));
    }
    
    logFinalOptimizationResult(bestTransform, bestMI);
    return bestTransform;
}

bool RegistrationStages::optimizeTranslation(const cv::Mat& ref, const cv::Mat& mov,
                                           cv::Mat& bestTransform, double& bestMI,
                                           double tStep, RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // Translation variations
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            cv::Mat candidate = bestTransform.clone();
            candidate.at<double>(0, 2) += dx * tStep;
            candidate.at<double>(1, 2) += dy * tStep;
            
            cv::Mat testAligned = metrics.applyTransform(mov, candidate);
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

bool RegistrationStages::optimizeRotation(const cv::Mat& ref, const cv::Mat& mov,
                                        cv::Mat& bestTransform, double& bestMI,
                                        double rStep, RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // Current transform parameters
    double a = bestTransform.at<double>(0, 0);
    double c = bestTransform.at<double>(1, 0);
    double currentScale = sqrt(a*a + c*c);
    double currentRotation = atan2(c, a);
    
    // Rotation variations
    for (int dr = -1; dr <= 1; dr += 2) {
        double newRotation = currentRotation + dr * rStep;
        double newCos = cos(newRotation);
        double newSin = sin(newRotation);
        
        cv::Mat candidate = bestTransform.clone();
        candidate.at<double>(0, 0) = currentScale * newCos;
        candidate.at<double>(0, 1) = -currentScale * newSin;
        candidate.at<double>(1, 0) = currentScale * newSin;
        candidate.at<double>(1, 1) = currentScale * newCos;
        
        cv::Mat testAligned = metrics.applyTransform(mov, candidate);
        double testMI = metrics.calculateMutualInformation(ref, testAligned);
        
        if (testMI > bestMI) {
            bestMI = testMI;
            bestTransform = candidate.clone();
            improved = true;
        }
    }
    
    return improved;
}

bool RegistrationStages::optimizeScale(const cv::Mat& ref, const cv::Mat& mov,
                                     cv::Mat& bestTransform, double& bestMI,
                                     double sStep, RegistrationMetricsCalculator& metrics) {
    bool improved = false;
    
    // Current transform parameters
    double a = bestTransform.at<double>(0, 0);
    double c = bestTransform.at<double>(1, 0);
    double currentScale = sqrt(a*a + c*c);
    double currentRotation = atan2(c, a);
    
    // Scale variations
    for (int ds = -1; ds <= 1; ds += 2) {
        double newScale = currentScale + ds * sStep;
        if (newScale > 0.5 && newScale < 2.0) {  // Reasonable scale bounds
            double cos_r = cos(currentRotation);
            double sin_r = sin(currentRotation);
            
            cv::Mat candidate = bestTransform.clone();
            candidate.at<double>(0, 0) = newScale * cos_r;
            candidate.at<double>(0, 1) = -newScale * sin_r;
            candidate.at<double>(1, 0) = newScale * sin_r;
            candidate.at<double>(1, 1) = newScale * cos_r;
            
            cv::Mat testAligned = metrics.applyTransform(mov, candidate);
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

void RegistrationStages::logFinalOptimizationResult(const cv::Mat& bestTransform, double bestMI) {
    // Final parameter summary
    double final_a = bestTransform.at<double>(0, 0);
    double final_c = bestTransform.at<double>(1, 0);
    double final_tx = bestTransform.at<double>(0, 2);
    double final_ty = bestTransform.at<double>(1, 2);
    
    double finalScale = sqrt(final_a*final_a + final_c*final_c);
    double finalRotation = atan2(final_c, final_a) * 180.0 / CV_PI;
    
    logProgress("    Final MI optimization result:");
    logProgress("      MI improved: " + to_string(bestMI));
    logProgress("      Scale: " + to_string(finalScale));
    logProgress("      Rotation: " + to_string(finalRotation) + " degrees");
    logProgress("      Translation: (" + to_string(final_tx) + ", " + to_string(final_ty) + ")");
}



} // namespace cell_registration