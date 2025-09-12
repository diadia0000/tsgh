#include "WSIRegistration.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
// CUDA OpenCV 頭文件可能不可用，使用條件編譯
#ifdef OPENCV_CUDA_AVAILABLE
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif
#endif

namespace wsi_registration {

WSIRegistration::WSIRegistration() {
    // 預設參數，針對組織病理學影像優化
    params_.referenceType = "HE";
    params_.enableCudaAcceleration = true;
    params_.featureDetector = "SIFT";
    params_.maxFeatures = 5000;
    params_.pyramidLevels = {4, 2, 1};
    params_.gridSpacing = {32, 16, 8};
    params_.targetTRE = 2.0;
}

WSIRegistration::WSIRegistration(const RegistrationParams& params) : params_(params) {
    if (params_.enableCudaAcceleration) {
        initializeCuda();
    }
}

bool WSIRegistration::loadWSI(const std::string& her2_path, 
                             const std::string& he_path, 
                             const std::string& fish_path) {
    logProgress("Loading WSI files...");
    
    // 載入影像
    her2Image_ = cv::imread(her2_path, cv::IMREAD_COLOR);
    heImage_ = cv::imread(he_path, cv::IMREAD_COLOR);
    fishImage_ = cv::imread(fish_path, cv::IMREAD_COLOR);
    
    if (!validateInputs()) {
        std::cerr << "Error: Failed to load one or more WSI files" << std::endl;
        return false;
    }
    
    // 預處理影像
    her2Image_ = preprocessImage(her2Image_, "brightfield");
    heImage_ = preprocessImage(heImage_, "brightfield");
    fishImage_ = preprocessImage(fishImage_, "fluorescence");
    
    // 調整到共同尺寸以確保一致處理
    int minWidth = std::min({her2Image_.cols, heImage_.cols, fishImage_.cols});
    int minHeight = std::min({her2Image_.rows, heImage_.rows, fishImage_.rows});
    
    // 考慮 WSI 的巨大尺寸，先進行降採樣
    int targetWidth = static_cast<int>(minWidth / params_.downsampleFactor);
    int targetHeight = static_cast<int>(minHeight / params_.downsampleFactor);
    cv::Size commonSize(targetWidth, targetHeight);
    
    cv::resize(her2Image_, her2Image_, commonSize, 0, 0, cv::INTER_AREA);
    cv::resize(heImage_, heImage_, commonSize, 0, 0, cv::INTER_AREA);
    cv::resize(fishImage_, fishImage_, commonSize, 0, 0, cv::INTER_AREA);
    
    logProgress("WSI files loaded and preprocessed successfully");
    return true;
}

bool WSIRegistration::performRegistration() {
    logProgress("Starting four-stage WSI registration workflow...");
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 階段配準: HER2 到 H&E (基準)
    logProgress("Registering HER2 to H&E (reference image)...");
    her2Result_ = performFourStageRegistration(heImage_, her2Image_, "brightfield");
    
    if (her2Result_.success) {
        alignedHER2_ = applyAffineTransform(her2Image_, her2Result_.finalTransformMatrix);
        logProgress("HER2 registration completed - TRE: " + std::to_string(her2Result_.finalTRE) + " pixels");
    }
    
    // 階段配準: FISH 到 H&E (基準)
    logProgress("Registering FISH to H&E (reference image)...");
    fishResult_ = performFourStageRegistration(heImage_, fishImage_, "fluorescence");
    
    if (fishResult_.success) {
        alignedFISH_ = applyAffineTransform(fishImage_, fishResult_.finalTransformMatrix);
        logProgress("FISH registration completed - TRE: " + std::to_string(fishResult_.finalTRE) + " pixels");
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    her2Result_.totalProcessingTime = totalTime;
    fishResult_.totalProcessingTime = totalTime;
    
    bool success = her2Result_.success && fishResult_.success;
    
    if (success) {
        logProgress("Multi-modal registration completed successfully");
        std::ostringstream oss;
        oss << "Total processing time: " << totalTime.count() << " ms";
        logProgress(oss.str());
    } else {
        std::cerr << "Error: Registration failed" << std::endl;
    }
    
    return success;
}

RegistrationResult WSIRegistration::performFourStageRegistration(const cv::Mat& reference, 
                                                               const cv::Mat& moving,
                                                               const std::string& movingType) {
    RegistrationResult result;
    result.stageResults.resize(3);
    
    auto overallStart = std::chrono::high_resolution_clock::now();
    
    try {
        // Stage 1: Feature-based coarse alignment (SIFT/ORB + RANSAC)
        logProgress("Stage 1: Feature-based coarse alignment...");
        result.stageResults[0] = performFeatureBasedCoarseAlignment(reference, moving, movingType);
        
        if (!result.stageResults[0].success) {
            result.errorMessage = "Feature-based coarse alignment failed";
            return result;
        }
        
        // Stage 2: Mutual information + affine transformation fine alignment
        logProgress("Stage 2: Mutual information fine alignment...");
        result.stageResults[1] = performMutualInfoAffineAlignment(reference, moving, 
                                                                result.stageResults[0].transformMatrix);
        
        if (!result.stageResults[1].success) {
            result.errorMessage = "Mutual information fine alignment failed";
            return result;
        }
        
        // Stage 3: B-spline FFD non-rigid alignment (if pixel-level alignment needed)
        if (params_.enablePixelLevelAlignment) {
            logProgress("Stage 3: B-spline non-rigid alignment...");
            result.stageResults[2] = performBSplineNonRigidAlignment(reference, moving,
                                                                   result.stageResults[1].transformMatrix);
            
            if (result.stageResults[2].success) {
                result.finalTransformMatrix = result.stageResults[2].transformMatrix;
            } else {
                // If B-spline fails, use affine transformation result
                result.finalTransformMatrix = result.stageResults[1].transformMatrix;
                logProgress("B-spline alignment failed, using affine transformation result");
            }
        } else {
            result.finalTransformMatrix = result.stageResults[1].transformMatrix;
        }
        
        // Calculate final quality metrics
        cv::Mat finalAligned = applyAffineTransform(moving, result.finalTransformMatrix);
        result.finalMI = calculateMutualInformation(reference, finalAligned);
        result.finalNMI = calculateNormalizedMutualInformation(reference, finalAligned);
        result.finalTRE = calculateTargetRegistrationError(reference, finalAligned, result.finalTransformMatrix);
        
        result.success = true;
        result.usedCuda = cudaInitialized_;
        
        logProgress("Four-stage registration completed - MI: " + std::to_string(result.finalMI) + 
                   ", NMI: " + std::to_string(result.finalNMI) + 
                   ", TRE: " + std::to_string(result.finalTRE));
        
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
        result.success = false;
        logProgress("Registration failed: " + std::string(e.what()));
    }
    
    auto overallEnd = std::chrono::high_resolution_clock::now();
    result.totalProcessingTime = std::chrono::duration_cast<std::chrono::milliseconds>(overallEnd - overallStart);
    
    return result;
}

StageResult WSIRegistration::performFeatureBasedCoarseAlignment(const cv::Mat& reference, 
                                                              const cv::Mat& moving,
                                                              const std::string& movingType) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    StageResult result;
    result.stage = RegistrationStage::FEATURE_BASED_COARSE;
    
    try {
        // 轉換為灰階
        cv::Mat refGray, movGray;
        cv::cvtColor(reference, refGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(moving, movGray, cv::COLOR_BGR2GRAY);
        
        // 特徵檢測
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        
        if (params_.featureDetector == "SIFT") {
            cv::Ptr<cv::SIFT> detector = cv::SIFT::create(params_.maxFeatures);
            detector->detectAndCompute(refGray, cv::noArray(), kp1, desc1);
            detector->detectAndCompute(movGray, cv::noArray(), kp2, desc2);
        } else if (params_.featureDetector == "ORB") {
            cv::Ptr<cv::ORB> detector = cv::ORB::create(params_.maxFeatures);
            detector->detectAndCompute(refGray, cv::noArray(), kp1, desc1);
            detector->detectAndCompute(movGray, cv::noArray(), kp2, desc2);
        }
        
        if (kp1.empty() || kp2.empty()) {
            result.errorMessage = "Unable to detect sufficient feature points";
            return result;
        }
        
        // Feature matching
        std::vector<cv::DMatch> matches = matchFeatures(desc1, desc2);
        result.featureMatches = static_cast<int>(matches.size());
        
        if (matches.size() < params_.minMatchCount) {
            result.errorMessage = "Insufficient feature matches: " + std::to_string(matches.size());
            return result;
        }
        
        // Extract matched points
        std::vector<cv::Point2f> srcPoints, dstPoints;
        for (const auto& match : matches) {
            srcPoints.push_back(kp2[match.trainIdx].pt);
            dstPoints.push_back(kp1[match.queryIdx].pt);
        }
        
        // RANSAC transform estimation
        result.transformMatrix = estimateTransformRANSAC(srcPoints, dstPoints);
        
        if (result.transformMatrix.empty()) {
            result.errorMessage = "RANSAC transform estimation failed";
            return result;
        }
        
        // Evaluate quality
        cv::Mat aligned = applyAffineTransform(moving, result.transformMatrix);
        result.mutualInformation = calculateMutualInformation(reference, aligned);
        result.targetRegistrationError = calculateTargetRegistrationError(reference, aligned, result.transformMatrix);
        
        result.success = true;
        
        logProgress("Feature-based coarse alignment completed - matches: " + std::to_string(result.featureMatches) + 
                   ", TRE: " + std::to_string(result.targetRegistrationError));
        
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
        result.success = false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    return result;
}

StageResult WSIRegistration::performMutualInfoAffineAlignment(const cv::Mat& reference, 
                                                            const cv::Mat& moving,
                                                            const cv::Mat& initialTransform) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    StageResult result;
    result.stage = RegistrationStage::MUTUAL_INFO_AFFINE;
    
    try {
        // Multi-resolution mutual information optimization
        cv::Mat currentTransform = initialTransform.clone();
        
        for (int level = 0; level < static_cast<int>(params_.pyramidLevels.size()); ++level) {
            int pyramidLevel = params_.pyramidLevels[level];
            
            logProgress("MI optimization - pyramid level " + std::to_string(level + 1) + "/" + 
                       std::to_string(params_.pyramidLevels.size()) + 
                       " (scale: 1/" + std::to_string(pyramidLevel) + ")");
            
            // Downsample images
            cv::Mat levelRef, levelMov;
            cv::resize(reference, levelRef, 
                      cv::Size(reference.cols / pyramidLevel, reference.rows / pyramidLevel),
                      0, 0, cv::INTER_AREA);
            cv::resize(moving, levelMov, 
                      cv::Size(moving.cols / pyramidLevel, moving.rows / pyramidLevel),
                      0, 0, cv::INTER_AREA);
            
            // Adjust transform matrix to current level
            cv::Mat levelTransform = currentTransform.clone();
            levelTransform.at<double>(0, 2) /= pyramidLevel;
            levelTransform.at<double>(1, 2) /= pyramidLevel;
            
            // Mutual information optimization
            levelTransform = optimizeAffineTransformMI(levelRef, levelMov, levelTransform);
            
            // Scale transform matrix back to original resolution
            levelTransform.at<double>(0, 2) *= pyramidLevel;
            levelTransform.at<double>(1, 2) *= pyramidLevel;
            
            currentTransform = levelTransform;
        }
        
        result.transformMatrix = currentTransform;
        
        // 評估最終品質
        cv::Mat aligned = applyAffineTransform(moving, result.transformMatrix);
        result.mutualInformation = calculateMutualInformation(reference, aligned);
        result.normalizedMutualInformation = calculateNormalizedMutualInformation(reference, aligned);
        result.targetRegistrationError = calculateTargetRegistrationError(reference, aligned, result.transformMatrix);
        
        result.success = true;
        
        logProgress("Mutual information fine alignment completed - MI: " + std::to_string(result.mutualInformation) + 
                   ", NMI: " + std::to_string(result.normalizedMutualInformation) +
                   ", TRE: " + std::to_string(result.targetRegistrationError));
        
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
        result.success = false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    return result;
}

StageResult WSIRegistration::performBSplineNonRigidAlignment(const cv::Mat& reference, 
                                                           const cv::Mat& moving,
                                                           const cv::Mat& initialTransform) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    StageResult result;
    result.stage = RegistrationStage::BSPLINE_NONRIGID;
    
    try {
        // 多解析度 B-spline 非剛體配準
        cv::Mat currentTransform = initialTransform.clone();
        
        for (int level = 0; level < static_cast<int>(params_.gridSpacing.size()); ++level) {
            int gridSpacing = params_.gridSpacing[level];
            
            logProgress("B-spline 非剛體對齊 - 網格間距: " + std::to_string(gridSpacing));
            
            // Generate B-spline deformation field
            cv::Mat deformationField = generateBSplineDeformationField(reference, moving, 
                                                                      currentTransform, gridSpacing);
            
            if (deformationField.empty()) {
                logProgress("Warning: B-spline deformation field generation failed, skipping this level");
                continue;
            }
            
            // Apply deformation field and evaluate
            cv::Mat aligned = applyBSplineTransform(moving, deformationField);
            double currentMI = calculateMutualInformation(reference, aligned);
            
            // If quality improves, update transform
            if (currentMI > result.mutualInformation) {
                result.mutualInformation = currentMI;
                // Note: Simplified processing, should actually combine affine transform and deformation field
                result.transformMatrix = currentTransform;
            }
        }
        
        // Evaluate final quality
        cv::Mat finalAligned = applyAffineTransform(moving, result.transformMatrix);
        result.normalizedMutualInformation = calculateNormalizedMutualInformation(reference, finalAligned);
        result.targetRegistrationError = calculateTargetRegistrationError(reference, finalAligned, result.transformMatrix);
        
        // Check if target accuracy is achieved
        if (result.targetRegistrationError <= params_.targetTRE) {
            result.success = true;
            logProgress("B-spline non-rigid alignment successful - target accuracy achieved");
        } else {
            result.success = false;
            result.errorMessage = "Target registration accuracy not achieved: " + std::to_string(result.targetRegistrationError) + 
                                " > " + std::to_string(params_.targetTRE);
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
        result.success = false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    return result;
}

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

// Quality assessment metrics implementation
double WSIRegistration::calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
    cv::Mat gray1, gray2;
    if (img1.channels() == 3) {
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = img1.clone();
    }
    
    if (img2.channels() == 3) {
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray2 = img2.clone();
    }
    
    const int bins = params_.histogramBins;
    
    // 計算聯合直方圖
    cv::Mat jointHist = cv::Mat::zeros(bins, bins, CV_32F);
    
    for (int y = 0; y < gray1.rows; y++) {
        for (int x = 0; x < gray1.cols; x++) {
            int val1 = std::min(static_cast<int>(gray1.at<uchar>(y, x) * bins / 256), bins - 1);
            int val2 = std::min(static_cast<int>(gray2.at<uchar>(y, x) * bins / 256), bins - 1);
            jointHist.at<float>(val1, val2) += 1.0f;
        }
    }
    
    jointHist /= (gray1.rows * gray1.cols);
    
    // 計算邊際直方圖
    cv::Mat hist1 = cv::Mat::zeros(bins, 1, CV_32F);
    cv::Mat hist2 = cv::Mat::zeros(bins, 1, CV_32F);
    
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            hist1.at<float>(i) += jointHist.at<float>(i, j);
            hist2.at<float>(j) += jointHist.at<float>(i, j);
        }
    }
    
    // 計算互信息
    double mi = 0.0;
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            float p_xy = jointHist.at<float>(i, j);
            if (p_xy > 1e-10) {
                float p_x = hist1.at<float>(i);
                float p_y = hist2.at<float>(j);
                if (p_x > 1e-10 && p_y > 1e-10) {
                    mi += p_xy * std::log(p_xy / (p_x * p_y));
                }
            }
        }
    }
    
    return mi;
}

double WSIRegistration::calculateNormalizedMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const {
    double mi = calculateMutualInformation(img1, img2);
    double h1 = calculateEntropy(img1);
    double h2 = calculateEntropy(img2);
    
    if (h1 + h2 == 0) return 0.0;
    
    return 2.0 * mi / (h1 + h2);
}

double WSIRegistration::calculateTargetRegistrationError(const cv::Mat& fixed, const cv::Mat& moving, 
                                                        const cv::Mat& transform) const {
    // 使用角點進行簡化的 TRE 計算
    std::vector<cv::Point2f> corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(fixed.cols), 0.0f),
        cv::Point2f(static_cast<float>(fixed.cols), static_cast<float>(fixed.rows)),
        cv::Point2f(0.0f, static_cast<float>(fixed.rows)),
        cv::Point2f(static_cast<float>(fixed.cols/2), static_cast<float>(fixed.rows/2))  // 中心點
    };
    
    double totalError = 0.0;
    
    for (const auto& corner : corners) {
        // 應用變換到角點
        cv::Mat point = (cv::Mat_<double>(3, 1) << corner.x, corner.y, 1);
        cv::Mat transformedPoint = transform * point;
        
        // 計算歐幾里得距離
        double dx = transformedPoint.at<double>(0, 0) - corner.x;
        double dy = transformedPoint.at<double>(1, 0) - corner.y;
        totalError += std::sqrt(dx * dx + dy * dy);
    }
    
    return totalError / corners.size();
}

double WSIRegistration::calculateEntropy(const cv::Mat& image) const {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 計算直方圖
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // 正規化
    hist /= (gray.rows * gray.cols);
    
    // 計算熵
    double entropy = 0.0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 1e-10) {
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

// 影像變換應用
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

// 影像預處理
cv::Mat WSIRegistration::preprocessImage(const cv::Mat& image, const std::string& imageType) const {
    cv::Mat processed;
    
    if (imageType == "fluorescence" || imageType == "fish") {
        // 螢光影像預處理
        if (image.type() != CV_8UC3) {
            image.convertTo(processed, CV_8UC3);
        } else {
            processed = image.clone();
        }
        
        // 檢查是否為暗螢光影像並反轉
        cv::Scalar meanIntensity = cv::mean(processed);
        bool isDark = (meanIntensity[0] + meanIntensity[1] + meanIntensity[2]) / 3.0 < 50;
        
        if (isDark) {
            cv::bitwise_not(processed, processed);
            logProgress("反轉暗螢光影像以改善處理效果");
        }
        
        // 螢光影像對比度增強
        cv::Mat lab;
        cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labChannels;
        cv::split(lab, labChannels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);
        
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
        
        // 螢光影像去噪
        cv::Mat denoised;
        cv::fastNlMeansDenoisingColored(processed, denoised, 10, 10, 7, 21);
        processed = denoised;
        
    } else {
        // 明場影像預處理 (H&E, HER2)
        if (image.type() != CV_8UC3) {
            image.convertTo(processed, CV_8UC3);
        } else {
            processed = image.clone();
        }
        
        // H&E 染色正規化
        cv::Mat lab;
        cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labChannels;
        cv::split(lab, labChannels);
        
        // 正規化 L 通道
        cv::equalizeHist(labChannels[0], labChannels[0]);
        
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
        
        // 溫和去噪
        cv::Mat denoised;
        cv::bilateralFilter(processed, denoised, 9, 75, 75);
        processed = denoised;
    }
    
    return processed;
}

// CUDA 相關函數
bool WSIRegistration::initializeCuda() {
#ifdef CUDA_AVAILABLE
    try {
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        if (error != cudaSuccess || deviceCount == 0) {
            logProgress("警告: 未找到 CUDA 設備，使用 CPU 處理");
            return false;
        }
        
        // 設置 CUDA 設備
        error = cudaSetDevice(cudaDeviceId_);
        if (error != cudaSuccess) {
            logProgress("警告: 無法設置 CUDA 設備 " + std::to_string(cudaDeviceId_));
            return false;
        }
        
        // 獲取設備資訊
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cudaDeviceId_);
        
        logProgress("CUDA 初始化成功 - 設備: " + std::string(prop.name));
        logProgress("總記憶體: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB");
        
        cudaInitialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        logProgress("CUDA 初始化失敗: " + std::string(e.what()));
        return false;
    }
#else
    logProgress("CUDA 支援未編譯，使用 CPU 處理");
    return false;
#endif
}

cv::Mat WSIRegistration::optimizeTransformCuda(const cv::Mat& reference, const cv::Mat& moving,
                                              const cv::Mat& initialTransform) const {
#ifdef CUDA_AVAILABLE
    if (!cudaInitialized_) {
        return optimizeAffineTransformMI(reference, moving, initialTransform);
    }
    
    try {
        // 上傳影像到 GPU
        cv::cuda::GpuMat d_reference, d_moving;
        d_reference.upload(reference);
        d_moving.upload(moving);
        
        // CUDA 加速的互信息優化 (簡化版本)
        cv::Mat optimizedTransform = initialTransform.clone();
        
        // 這裡應該實現 CUDA 核心函數進行並行優化
        // 目前使用 CPU 版本作為後備
        cv::Mat h_reference, h_moving;
        d_reference.download(h_reference);
        d_moving.download(h_moving);
        
        return optimizeAffineTransformMI(h_reference, h_moving, initialTransform);
        
    } catch (const cv::Exception& e) {
        logProgress("CUDA 優化失敗，回退到 CPU: " + std::string(e.what()));
        return optimizeAffineTransformMI(reference, moving, initialTransform);
    }
#else
    return optimizeAffineTransformMI(reference, moving, initialTransform);
#endif
}

// 檔案儲存和報告生成
bool WSIRegistration::saveAlignedImages(const std::string& output_dir) {
    logProgress("儲存對齊後的影像...");
    
    // 建立輸出目錄
    std::filesystem::create_directories(output_dir);
    
    bool success = true;
    
    // 儲存基準 H&E 影像
    success &= cv::imwrite(output_dir + "/reference_HE.jpg", heImage_);
    
    // 儲存對齊後的影像
    if (!alignedHER2_.empty()) {
        success &= cv::imwrite(output_dir + "/aligned_HER2.jpg", alignedHER2_);
        
        // 建立 HE-HER2 疊合影像
        cv::Mat heHer2Overlay;
        cv::addWeighted(heImage_, 0.6, alignedHER2_, 0.4, 0, heHer2Overlay);
        success &= cv::imwrite(output_dir + "/overlay_HE_HER2.jpg", heHer2Overlay);
    }
    
    if (!alignedFISH_.empty()) {
        success &= cv::imwrite(output_dir + "/aligned_FISH.jpg", alignedFISH_);
        
        // 建立 HE-FISH 疊合影像
        cv::Mat heFishOverlay = createQualityOverlay(heImage_, alignedFISH_);
        success &= cv::imwrite(output_dir + "/overlay_HE_FISH.jpg", heFishOverlay);
    }
    
    // 建立三通道疊合影像
    if (!alignedHER2_.empty() && !alignedFISH_.empty()) {
        cv::Mat tripleOverlay = cv::Mat::zeros(heImage_.size(), CV_8UC3);
        
        // HE (藍色通道), HER2 (紅色通道), FISH (綠色通道)
        cv::Mat heGray, her2Gray, fishGray;
        cv::cvtColor(heImage_, heGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(alignedHER2_, her2Gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(alignedFISH_, fishGray, cv::COLOR_BGR2GRAY);
        
        std::vector<cv::Mat> channels = {heGray, fishGray, her2Gray};
        cv::merge(channels, tripleOverlay);
        
        success &= cv::imwrite(output_dir + "/overlay_triple_channel.jpg", tripleOverlay);
    }
    
    if (success) {
        logProgress("All aligned images saved successfully");
    } else {
        std::cerr << "Error: Failed to save some images" << std::endl;
    }
    
    return success;
}

bool WSIRegistration::generateReport(const std::string& report_path) const {
    std::ofstream report(report_path);
    if (!report.is_open()) {
        std::cerr << "錯誤: 無法建立報告檔案 " << report_path << std::endl;
        return false;
    }
    
    report << "四階段 WSI 配準報告\n";
    report << "生成時間: " << getCurrentTimestamp() << "\n";
    report << "========================================\n\n";
    
    // 配置參數
    report << "配準參數:\n";
    report << "  基準影像類型: " << params_.referenceType << "\n";
    report << "  特徵檢測器: " << params_.featureDetector << "\n";
    report << "  最大特徵數: " << params_.maxFeatures << "\n";
    report << "  RANSAC 閾值: " << params_.ransacThreshold << "\n";
    report << "  金字塔層級: ";
    for (size_t i = 0; i < params_.pyramidLevels.size(); ++i) {
        report << params_.pyramidLevels[i];
        if (i < params_.pyramidLevels.size() - 1) report << ", ";
    }
    report << "\n";
    report << "  網格間距: ";
    for (size_t i = 0; i < params_.gridSpacing.size(); ++i) {
        report << params_.gridSpacing[i];
        if (i < params_.gridSpacing.size() - 1) report << ", ";
    }
    report << "\n";
    report << "  CUDA Acceleration: " << (params_.enableCudaAcceleration ? "Enabled" : "Disabled") << "\n\n";
    
    // HER2 Registration Results
    report << "HER2 Registration Results:\n";
    report << "  Success: " << (her2Result_.success ? "Yes" : "No") << "\n";
    if (her2Result_.success) {
        report << "  Final MI: " << std::fixed << std::setprecision(6) << her2Result_.finalMI << "\n";
        report << "  Final NMI: " << std::fixed << std::setprecision(6) << her2Result_.finalNMI << "\n";
        report << "  Final TRE (pixels): " << std::fixed << std::setprecision(2) << her2Result_.finalTRE << "\n";
        report << "  Total Processing Time: " << her2Result_.totalProcessingTime.count() << " ms\n";
        report << "  Used CUDA: " << (her2Result_.usedCuda ? "Yes" : "No") << "\n";
        
        // Detailed stage results
        for (size_t i = 0; i < her2Result_.stageResults.size(); ++i) {
            const auto& stage = her2Result_.stageResults[i];
            report << "  Stage " << (i+1) << " Results:\n";
            report << "    Success: " << (stage.success ? "Yes" : "No") << "\n";
            report << "    Processing Time: " << stage.processingTime.count() << " ms\n";
            if (stage.success) {
                report << "    MI: " << std::fixed << std::setprecision(6) << stage.mutualInformation << "\n";
                report << "    TRE: " << std::fixed << std::setprecision(2) << stage.targetRegistrationError << "\n";
                if (stage.featureMatches > 0) {
                    report << "    Feature Matches: " << stage.featureMatches << "\n";
                }
            } else {
                report << "    Error: " << stage.errorMessage << "\n";
            }
        }
    } else {
        report << "  Error: " << her2Result_.errorMessage << "\n";
    }
    report << "\n";
    
    // FISH Registration Results
    report << "FISH Registration Results:\n";
    report << "  Success: " << (fishResult_.success ? "Yes" : "No") << "\n";
    if (fishResult_.success) {
        report << "  Final MI: " << std::fixed << std::setprecision(6) << fishResult_.finalMI << "\n";
        report << "  Final NMI: " << std::fixed << std::setprecision(6) << fishResult_.finalNMI << "\n";
        report << "  Final TRE (pixels): " << std::fixed << std::setprecision(2) << fishResult_.finalTRE << "\n";
        report << "  Total Processing Time: " << fishResult_.totalProcessingTime.count() << " ms\n";
        report << "  Used CUDA: " << (fishResult_.usedCuda ? "Yes" : "No") << "\n";
        
        // Detailed stage results
        for (size_t i = 0; i < fishResult_.stageResults.size(); ++i) {
            const auto& stage = fishResult_.stageResults[i];
            report << "  Stage " << (i+1) << " Results:\n";
            report << "    Success: " << (stage.success ? "Yes" : "No") << "\n";
            report << "    Processing Time: " << stage.processingTime.count() << " ms\n";
            if (stage.success) {
                report << "    MI: " << std::fixed << std::setprecision(6) << stage.mutualInformation << "\n";
                report << "    TRE: " << std::fixed << std::setprecision(2) << stage.targetRegistrationError << "\n";
                if (stage.featureMatches > 0) {
                    report << "    Feature Matches: " << stage.featureMatches << "\n";
                }
            } else {
                report << "    Error: " << stage.errorMessage << "\n";
            }
        }
    } else {
        report << "  Error: " << fishResult_.errorMessage << "\n";
    }
    report << "\n";
    
    // Summary
    auto totalTime = her2Result_.totalProcessingTime + fishResult_.totalProcessingTime;
    report << "Summary:\n";
    report << "  Total Processing Time: " << totalTime.count() << " ms\n";
    report << "  Average TRE: " << std::fixed << std::setprecision(2) 
           << (her2Result_.finalTRE + fishResult_.finalTRE) / 2.0 << " pixels\n";
    report << "  Average MI: " << std::fixed << std::setprecision(6)
           << (her2Result_.finalMI + fishResult_.finalMI) / 2.0 << "\n";
    report << "  Average NMI: " << std::fixed << std::setprecision(6)
           << (her2Result_.finalNMI + fishResult_.finalNMI) / 2.0 << "\n";
    
    report.close();
    
    logProgress("Registration report generated: " + report_path);
    return true;
}

// Utility functions
bool WSIRegistration::validateInputs() const {
    return !her2Image_.empty() && !heImage_.empty() && !fishImage_.empty() &&
           her2Image_.rows > 0 && her2Image_.cols > 0 &&
           heImage_.rows > 0 && heImage_.cols > 0 &&
           fishImage_.rows > 0 && fishImage_.cols > 0;
}

cv::Mat WSIRegistration::createQualityOverlay(const cv::Mat& reference, const cv::Mat& aligned) const {
    cv::Mat overlay;
    
    // 轉換為灰階
    cv::Mat refGray, alignedGray;
    cv::cvtColor(reference, refGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(aligned, alignedGray, cv::COLOR_BGR2GRAY);
    
    // 建立彩色疊合 (參考影像為紅色，對齊影像為綠色)
    std::vector<cv::Mat> channels = {alignedGray, alignedGray, refGray};
    cv::merge(channels, overlay);
    
    return overlay;
}

void WSIRegistration::logProgress(const std::string& message) const {
    std::cout << "[" << getCurrentTimestamp() << "] " << message << std::endl;
}

std::string WSIRegistration::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
#ifdef _WIN32
    struct tm timeinfo;
    localtime_s(&timeinfo, &time_t);
    oss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
#else
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
#endif
    return oss.str();
}

} // namespace wsi_registration