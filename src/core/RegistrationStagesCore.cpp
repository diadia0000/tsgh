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
    
    // 改進的灰階預處理 - 針對每張圖片進行像素級優化
    Mat refEnhanced, movEnhanced;
    
    // 使用CLAHE增強對比度
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(refGray, refEnhanced);
    clahe->apply(movGray, movEnhanced);
    
    // 高斯濾波去噪但保留邊緣
    GaussianBlur(refEnhanced, refEnhanced, Size(3, 3), 0.8);
    GaussianBlur(movEnhanced, movEnhanced, Size(3, 3), 0.8);
    
    // 銳化增強邊緣特徵
    Mat kernel = (Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(refEnhanced, refEnhanced, -1, kernel);
    filter2D(movEnhanced, movEnhanced, -1, kernel);
    
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
    
    // 更嚴格的比例測試以提高匹配品質
    vector<DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match.size() == 2 && match[0].distance < 0.6 * match[1].distance) {  // 更嚴格的比例
            goodMatches.push_back(match[0]);
        }
    }
    
    // 如果匹配太少，稍微放寬條件
    if (goodMatches.size() < 50) {
        goodMatches.clear();
        for (const auto& match : knnMatches) {
            if (match.size() == 2 && match[0].distance < 0.75 * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
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
        // Use RANSAC to estimate robust affine transformation with stricter parameters
        Mat mask;
        transform = estimateAffine2D(srcPoints, dstPoints, mask, 
                                       RANSAC, 3.0, 5000, 0.995, 50);  // 更嚴格的參數
        
        if (transform.empty() || transform.rows != 2 || transform.cols != 3) {
            logProgress("    Affine estimation failed, trying partial affine");
            
            // Fallback: estimate similarity transform (translation + rotation + uniform scaling)
            transform = estimateAffinePartial2D(srcPoints, dstPoints, mask, 
                                                   RANSAC, 3.0, 5000, 0.995, 50);
        }
        
        // 驗證變換參數是否合理 - 放寬限制但避免極端值
        bool isValidTransform = false;
        if (!transform.empty() && transform.rows == 2 && transform.cols == 3) {
            double a = transform.at<double>(0, 0);
            double b = transform.at<double>(0, 1);
            double c = transform.at<double>(1, 0);
            double d = transform.at<double>(1, 1);
            double tx = transform.at<double>(0, 2);
            double ty = transform.at<double>(1, 2);
            
            double scaleX = sqrt(a*a + c*c);
            double scaleY = sqrt(b*b + d*d);
            double rotation = atan2(c, a) * 180.0 / CV_PI;
            
            // 放寬參數範圍，但仍避免極端變換
            bool validScale = (scaleX > 0.1 && scaleX < 20.0) && (scaleY > 0.1 && scaleY < 20.0);  // 大幅放寬縮放限制
            bool validRotation = abs(rotation) < 180.0;  // 允許更大的旋轉角度
            bool validTranslation = abs(tx) < 200000 && abs(ty) < 200000;  // 極大放寬平移限制以適應WSI大偏移
            
            // 記錄實際參數值以便調試
            logProgress("    Checking transform validity:");
            logProgress("      Scale: (" + to_string(scaleX) + ", " + to_string(scaleY) + ") - " + (validScale ? "OK" : "INVALID"));
            logProgress("      Rotation: " + to_string(rotation) + " degrees - " + (validRotation ? "OK" : "INVALID"));
            logProgress("      Translation: (" + to_string(tx) + ", " + to_string(ty) + ") - " + (validTranslation ? "OK" : "INVALID"));
            
            isValidTransform = validScale && validRotation && validTranslation;
        }
        
        if (!isValidTransform) {
            logProgress("    Transform parameters unreasonable, using identity");
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
    
    // 合理的參數檢查，避免極端變換但允許必要的調整
    bool validScale = (scaleX > 0.1 && scaleX < 20.0) && (scaleY > 0.1 && scaleY < 20.0);  // 大幅放寬縮放限制
    bool validRotation = abs(rotation) < 360.0;  // 允許較大的旋轉角度
    bool validTranslation = abs(tx) < 200000 && abs(ty) < 200000;  // 極大放寬平移限制以適應WSI大偏移
    
    logProgress("    Transform validation:");
    logProgress("      Scale: (" + to_string(scaleX) + ", " + to_string(scaleY) + ")");
    logProgress("      Rotation: " + to_string(rotation) + " degrees");
    logProgress("      Translation: (" + to_string(tx) + ", " + to_string(ty) + ")");
    
    if (!validScale || !validRotation || !validTranslation) {
        logProgress("    Transform parameters out of reasonable range - using identity");
    } else {
        logProgress("    Transform parameters within acceptable range");
        
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
    
    // 改進的灰階預處理用於互信息計算
    Mat refGray, movGray;
    
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
    
    // 使用CLAHE增強對比度以提高互信息計算精度
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(refGray, refGray);
    clahe->apply(movGray, movGray);
    
    // Multi-resolution optimization for better convergence
    Mat bestTransform = optimized.clone();
    double bestMI = 0.0;
    
    // Calculate initial MI using enhanced grayscale images
    RegistrationMetricsCalculator metrics;
    Mat aligned = metrics.applyTransform(movGray, optimized);
    bestMI = metrics.calculateMutualInformation(refGray, aligned);
    logProgress("    Initial MI (enhanced): " + to_string(bestMI));
    
    return performMultiParameterOptimization(refGray, movGray, bestTransform, bestMI);
}

// Optimization functions moved to RegistrationStagesOptimization.cpp

void RegistrationStages::logProgress(const std::string& message) {
    cout << "[" << chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch()).count() % 100000 
              << "] " << message << endl;
}

} // namespace cell_registration