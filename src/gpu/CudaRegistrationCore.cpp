#include "wsi/CudaRegistration.h"
#include "wsi/CudaUtils.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace cuda_registration {

// CudaRegistrationEngine implementation
CudaRegistrationEngine::CudaRegistrationEngine(const CudaRegistrationConfig& config) 
    : config_(config) {
    similarityMetrics_ = std::make_unique<CudaSimilarityMetrics>(config);
    imageProcessor_ = std::make_unique<CudaImageProcessor>(config);
}

bool CudaRegistrationEngine::initialize() {
    // MVP: Simple initialization check
    if (!CudaUtils::isDeviceAvailable(config_.deviceId)) {
        std::cerr << "Warning: CUDA device " << config_.deviceId 
                  << " not available, using CPU fallback" << std::endl;
    }
    
    isInitialized_ = true;
    return true;
}

void CudaRegistrationEngine::shutdown() {
    isInitialized_ = false;
}

cv::Mat CudaRegistrationEngine::registerImages(const cv::Mat& fixed, const cv::Mat& moving) {
    if (!isInitialized_) {
        std::cerr << "Error: CUDA registration engine not initialized" << std::endl;
        return cv::Mat::eye(3, 3, CV_64F);
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    cv::Mat transform = optimizeTransform(fixed, moving);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    lastRegistrationTime_ = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    return transform;
}

cv::Mat CudaRegistrationEngine::multiResolutionRegistration(const cv::Mat& fixed, const cv::Mat& moving,
                                                           const std::vector<int>& pyramidLevels) {
    // Build pyramids
    std::vector<cv::Mat> fixedPyramid = imageProcessor_->buildGaussianPyramid(fixed, 
                                                                             static_cast<int>(pyramidLevels.size()));
    std::vector<cv::Mat> movingPyramid = imageProcessor_->buildGaussianPyramid(moving, 
                                                                              static_cast<int>(pyramidLevels.size()));
    
    cv::Mat accumulatedTransform = cv::Mat::eye(3, 3, CV_64F);
    
    // Process each pyramid level
    for (size_t i = 0; i < pyramidLevels.size() && i < fixedPyramid.size(); ++i) {
        cv::Mat levelTransform = registerImages(fixedPyramid[i], movingPyramid[i]);
        
        // Scale transform for current level
        double scale = std::pow(2.0, static_cast<double>(i));
        levelTransform.at<double>(0, 2) *= scale;
        levelTransform.at<double>(1, 2) *= scale;
        
        // Accumulate transforms
        accumulatedTransform = levelTransform * accumulatedTransform;
    }
    
    return accumulatedTransform;
}

std::vector<cv::Mat> CudaRegistrationEngine::registerImageBatch(const cv::Mat& fixed,
                                                               const std::vector<cv::Mat>& movingImages) {
    std::vector<cv::Mat> transforms;
    transforms.reserve(movingImages.size());
    
    // MVP: Sequential processing (in production, parallel GPU processing)
    for (const auto& moving : movingImages) {
        transforms.push_back(registerImages(fixed, moving));
    }
    
    return transforms;
}

size_t CudaRegistrationEngine::getGpuMemoryUsage() const {
    // MVP: Return simulated memory usage
    return config_.maxGpuMemory / 2; // Assume 50% usage
}

cv::Mat CudaRegistrationEngine::optimizeTransform(const cv::Mat& fixed, const cv::Mat& moving) {
    return gradientDescentOptimization(fixed, moving);
}

cv::Mat CudaRegistrationEngine::gradientDescentOptimization(const cv::Mat& fixed, const cv::Mat& moving) {
    // MVP: Simplified gradient descent using feature matching
    
    // Detect features
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    detector->detectAndCompute(fixed, cv::Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(moving, cv::Mat(), keypoints2, descriptors2);
    
    if (keypoints1.empty() || keypoints2.empty()) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    
    // Match features
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    if (matches.size() < 4) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    
    // Extract matched points
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    
    // Estimate transform
    cv::Mat transform = cv::estimateAffinePartial2D(points2, points1, cv::noArray(), 
                                                   cv::RANSAC, 3.0);
    
    if (transform.empty()) {
        return cv::Mat::eye(3, 3, CV_64F);
    }
    
    // Convert to 3x3 homogeneous matrix
    cv::Mat homogeneous = cv::Mat::eye(3, 3, CV_64F);
    transform.copyTo(homogeneous(cv::Rect(0, 0, 3, 2)));
    
    return homogeneous;
}

} // namespace cuda_registration