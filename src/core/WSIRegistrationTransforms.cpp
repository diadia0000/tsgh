#include "wsi/WSIRegistration.h"
#include <algorithm>
#include <cmath>

namespace wsi_registration {

std::vector<cv::DMatch> WSIRegistration::matchFeatures(const cv::Mat& desc1, 
                                                      const cv::Mat& desc2) const {
    std::vector<cv::DMatch> goodMatches;
    
    if (desc1.empty() || desc2.empty()) {
        return goodMatches;
    }
    
    // 使用 FLANN 匹配器提高效能
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    
    try {
        matcher.knnMatch(desc1, desc2, knnMatches, 2);
    } catch (const cv::Exception&) {
        logProgress("Warning: FLANN matching failed, falling back to brute force");
        cv::BFMatcher bfMatcher;
        bfMatcher.knnMatch(desc1, desc2, knnMatches, 2);
    }
    
    // Lowe's ratio test
    for (const auto& match : knnMatches) {
        if (match.size() == 2 && match[0].distance < params_.ratioTestThreshold * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    
    return goodMatches;
}

cv::Mat WSIRegistration::estimateTransformRANSAC(const std::vector<cv::Point2f>& srcPoints,
                                                 const std::vector<cv::Point2f>& dstPoints) const {
    if (srcPoints.size() < 4 || dstPoints.size() < 4) {
        return cv::Mat();
    }
    
    // Use RANSAC to estimate affine transformation
    cv::Mat transform = cv::estimateAffine2D(srcPoints, dstPoints, cv::noArray(), 
                                           cv::RANSAC, params_.ransacThreshold, 
                                           2000, 0.99);
    
    if (transform.empty()) {
        return cv::Mat();
    }
    
    // Convert to 3x3 homogeneous matrix
    cv::Mat homogeneous = cv::Mat::eye(3, 3, CV_64F);
    transform.copyTo(homogeneous(cv::Rect(0, 0, 3, 2)));
    
    return homogeneous;
}

cv::Mat WSIRegistration::optimizeAffineTransformMI(const cv::Mat& reference, 
                                                  const cv::Mat& moving,
                                                  const cv::Mat& initialTransform) const {
    cv::Mat optimizedTransform = initialTransform.clone();
    
    if (cudaInitialized_ && params_.enableCudaAcceleration) {
        // 使用 CUDA 加速優化
        return optimizeTransformCuda(reference, moving, initialTransform);
    }
    
    // CPU 版本的簡化梯度下降優化
    double currentMI = calculateMutualInformation(reference, 
                                                 applyAffineTransform(moving, optimizedTransform));
    double learningRate = 0.1;
    int maxIterations = params_.maxIterationsAffine;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // 小幅擾動進行梯度估計
        double delta = 0.5;
        bool improved = false;
        
        // 測試平移擾動
        for (int param = 0; param < 2; ++param) {  // 只優化平移參數
            cv::Mat testTransform = optimizedTransform.clone();
            testTransform.at<double>(param, 2) += delta;
            
            double testMI = calculateMutualInformation(reference, 
                                                     applyAffineTransform(moving, testTransform));
            
            if (testMI > currentMI) {
                optimizedTransform.at<double>(param, 2) += learningRate * delta;
                currentMI = testMI;
                improved = true;
            }
        }
        
        if (!improved) {
            learningRate *= 0.9;  // 減少學習率
            if (learningRate < 0.01) break;
        }
        
        // 收斂檢查
        if (iter > 10 && improved == false) {
            break;
        }
    }
    
    return optimizedTransform;
}

cv::Mat WSIRegistration::generateBSplineDeformationField(const cv::Mat& reference,
                                                        const cv::Mat& moving,
                                                        const cv::Mat& initialTransform,
                                                        int gridSpacing) const {
    // 簡化的 B-spline 變形場生成
    // 實際實現需要更複雜的 B-spline 基函數計算
    
    cv::Mat deformationField = cv::Mat::zeros(reference.rows, reference.cols, CV_32FC2);
    
    // 創建控制點網格
    int gridRows = reference.rows / gridSpacing + 1;
    int gridCols = reference.cols / gridSpacing + 1;
    
    // 簡化實現: 基於局部互信息計算控制點位移
    for (int i = 0; i < gridRows - 1; ++i) {
        for (int j = 0; j < gridCols - 1; ++j) {
            int y = i * gridSpacing;
            int x = j * gridSpacing;
            
            if (y + gridSpacing < reference.rows && x + gridSpacing < reference.cols) {
                cv::Rect roi(x, y, gridSpacing, gridSpacing);
                cv::Mat refPatch = reference(roi);
                cv::Mat movPatch = moving(roi);
                
                // 應用當前變換到移動影像塊
                cv::Mat transformedPatch = applyAffineTransform(movPatch, initialTransform);
                
                // 計算局部最佳位移 (簡化版本)
                cv::Point2f displacement(0, 0);
                double bestMI = calculateMutualInformation(refPatch, transformedPatch);
                
                // 搜索小範圍位移
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        cv::Mat shiftedPatch;
                        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
                        cv::warpAffine(transformedPatch, shiftedPatch, M, transformedPatch.size());
                        
                        double mi = calculateMutualInformation(refPatch, shiftedPatch);
                        if (mi > bestMI) {
                            bestMI = mi;
                            displacement = cv::Point2f(static_cast<float>(dx), static_cast<float>(dy));
                        }
                    }
                }
                
                // 將位移應用到變形場
                for (int py = y; py < y + gridSpacing && py < deformationField.rows; ++py) {
                    for (int px = x; px < x + gridSpacing && px < deformationField.cols; ++px) {
                        deformationField.at<cv::Vec2f>(py, px) = cv::Vec2f(displacement.x, displacement.y);
                    }
                }
            }
        }
    }
    
    return deformationField;
}

cv::Mat WSIRegistration::applyAffineTransform(const cv::Mat& image, const cv::Mat& transform) const {
    cv::Mat result;
    cv::Mat affineTransform = transform(cv::Rect(0, 0, 3, 2));  // 提取 2x3 仿射矩陣
    cv::warpAffine(image, result, affineTransform, image.size(),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat WSIRegistration::applyBSplineTransform(const cv::Mat& image, const cv::Mat& deformationField) const {
    cv::Mat result;
    cv::remap(image, result, deformationField, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat WSIRegistration::optimizeTransformCuda(const cv::Mat& reference, 
                                              const cv::Mat& moving,
                                              const cv::Mat& initialTransform) const {
    // Placeholder for CUDA optimization implementation
    // In a full implementation, this would use CUDA kernels for acceleration
    logProgress("CUDA optimization not fully implemented, using CPU fallback");
    return optimizeAffineTransformMI(reference, moving, initialTransform);
}

} // namespace wsi_registration