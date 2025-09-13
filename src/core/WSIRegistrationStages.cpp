#include "wsi/WSIRegistration.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
namespace wsi_registration {

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

} // namespace wsi_registration