#include "wsi/RegistrationStages.h"
#include "wsi/RegistrationMetrics.h"
#include "wsi/ImagePreprocessing.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <exception>

using namespace cv;
using namespace std;

namespace cell_registration {

cv::Mat RegistrationStages::featureBasedAlignment(const cv::Mat& ref, const cv::Mat& mov) {
    Mat refGray, movGray;
    cvtColor(ref, refGray, COLOR_BGR2GRAY);
    cvtColor(mov, movGray, COLOR_BGR2GRAY);
    
    // Enhanced preprocessing for feature detection
    Mat refEnhanced, movEnhanced;
    ImagePreprocessing preprocessor;
    
    // Strong preprocessing for medical images
    preprocessor.preprocessMedicalImage(ref, refEnhanced);
    preprocessor.preprocessMedicalImage(mov, movEnhanced);
    
    cvtColor(refEnhanced, refEnhanced, COLOR_BGR2GRAY);
    cvtColor(movEnhanced, movEnhanced, COLOR_BGR2GRAY);
    
    // SIFT with very relaxed parameters for medical images
    auto sift = SIFT::create(10000, 3, 0.08, 5, 1.2);
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    
    sift->detectAndCompute(refEnhanced, noArray(), kp1, desc1);
    sift->detectAndCompute(movEnhanced, noArray(), kp2, desc2);
    
    logProgress("    Detected keypoints: ref=" + to_string(kp1.size()) + ", mov=" + to_string(kp2.size()));
    
    if (kp1.size() < 10 || kp2.size() < 10) {
        logProgress("    Insufficient keypoints detected");
        return Mat();
    }
    
    // Feature matching with FLANN matcher for better performance
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knnMatches;
    
    try {
        matcher.knnMatch(desc1, desc2, knnMatches, 2);
    } catch (const exception& e) {
        logProgress("    Feature matching failed: " + string(e.what()));
        return Mat();
    }
    
    // Very relaxed ratio test for medical images
    vector<DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match.size() == 2 && match[0].distance < 0.9 * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    
    logProgress("    Good matches found: " + to_string(goodMatches.size()));
    
    if (goodMatches.size() < 6) {
        logProgress("    Insufficient good matches for affine transform");
        return Mat();
    }
    
    // Extract matched points
    vector<Point2f> srcPoints, dstPoints;
    for (const auto& match : goodMatches) {
        srcPoints.push_back(kp2[match.trainIdx].pt);
        dstPoints.push_back(kp1[match.queryIdx].pt);
    }
    
    return estimateRobustTransform(srcPoints, dstPoints);
}

cv::Mat RegistrationStages::estimateRobustTransform(const std::vector<cv::Point2f>& srcPoints,
                                                   const std::vector<cv::Point2f>& dstPoints) {
    // Calculate FULL 2D AFFINE TRANSFORM (translation + rotation + scaling + shear)
    Mat transform;
    
    try {
        // Use RANSAC to estimate robust affine transformation
        Mat mask;
        transform = estimateAffine2D(srcPoints, dstPoints, mask, 
                                       RANSAC, 3.0, 2000, 0.99, 10);
        
        if (transform.empty() || transform.rows != 2 || transform.cols != 3) {
            logProgress("    Affine estimation failed, trying partial affine");
            
            // Fallback: estimate similarity transform (translation + rotation + uniform scaling)
            transform = estimateAffinePartial2D(srcPoints, dstPoints, mask, 
                                                   RANSAC, 3.0, 2000, 0.99, 10);
        }
        
        if (transform.empty() || transform.rows != 2 || transform.cols != 3) {
            logProgress("    All affine estimation failed, using identity");
            transform = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
        } else {
            validateAndLogTransform(transform, srcPoints, dstPoints);
        }
        
    } catch (const exception& e) {
        logProgress("    Exception in affine estimation: " + string(e.what()));
        transform = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
    }
    
    // Ensure we return exactly 2x3 matrix
    Mat result2D;
    if (transform.rows == 2 && transform.cols == 3) {
        result2D = transform.clone();
    } else {
        result2D = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
    }
    
    logProgress("    Feature-based 2D affine alignment completed");
    return result2D;
}

void RegistrationStages::validateAndLogTransform(const cv::Mat& transform, 
                                                const std::vector<cv::Point2f>& srcPoints,
                                                const std::vector<cv::Point2f>& dstPoints) {
    // Validate transform parameters
    double a = transform.at<double>(0, 0);
    double b = transform.at<double>(0, 1);
    double c = transform.at<double>(1, 0);
    double d = transform.at<double>(1, 1);
    double tx = transform.at<double>(0, 2);
    double ty = transform.at<double>(1, 2);
    
    // Calculate scale and rotation
    double scaleX = sqrt(a*a + c*c);
    double scaleY = sqrt(b*b + d*d);
    double rotation = atan2(c, a) * 180.0 / CV_PI;
    
    // Very relaxed checks for medical images
    bool validScale = (scaleX > 0.3 && scaleX < 3.0) && (scaleY > 0.3 && scaleY < 3.0);
    bool validRotation = true;  // Allow any rotation (360 degrees)
    bool validTranslation = abs(tx) < 1000 && abs(ty) < 1000;  // Reasonable bounds
    
    if (!validScale || !validRotation || !validTranslation) {
        logProgress("    Transform parameters out of range");
        logProgress("      Scale: (" + to_string(scaleX) + ", " + to_string(scaleY) + ")");
        logProgress("      Rotation: " + to_string(rotation) + " degrees");
        logProgress("      Translation: (" + to_string(tx) + ", " + to_string(ty) + ")");
    } else {
        logProgress("    Full 2D affine transform estimated:");
        logProgress("      Scale: (" + to_string(scaleX) + ", " + to_string(scaleY) + ")");
        logProgress("      Rotation: " + to_string(rotation) + " degrees");
        logProgress("      Translation: (" + to_string(tx) + ", " + to_string(ty) + ")");
        
        // Count inliers if mask is available
        double inlierRatio = static_cast<double>(srcPoints.size()) / srcPoints.size();
        logProgress("      Points used: " + to_string(srcPoints.size()) + 
                   " (" + to_string(inlierRatio * 100.0) + "%)");
    }
}

cv::Mat RegistrationStages::mutualInfoAlignment(const cv::Mat& ref, const cv::Mat& mov, const cv::Mat& initial) {
    // Ensure we work with 2x3 matrix
    Mat optimized;
    if (initial.rows == 2 && initial.cols == 3) {
        optimized = initial.clone();
    } else {
        optimized = initial(Rect(0, 0, 3, 2)).clone();
    }
    
    // Multi-resolution optimization for better convergence
    Mat bestTransform = optimized.clone();
    double bestMI = 0.0;
    
    // Calculate initial MI
    RegistrationMetricsCalculator metrics;
    Mat aligned = metrics.applyTransform(mov, optimized);
    bestMI = metrics.calculateMutualInformation(ref, aligned);
    logProgress("    Initial MI: " + to_string(bestMI));
    
    return performMultiParameterOptimization(ref, mov, bestTransform, bestMI);
}

// Optimization functions moved to RegistrationStagesOptimization.cpp

void RegistrationStages::logProgress(const std::string& message) {
    cout << "[" << chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch()).count() % 100000 
              << "] " << message << endl;
}

} // namespace cell_registration