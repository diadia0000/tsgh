#include "wsi/WSIRegistration.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
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

bool WSIRegistration::validateInputs() const {
    return !her2Image_.empty() && !heImage_.empty() && !fishImage_.empty();
}

void WSIRegistration::logProgress(const std::string& message) const {
    std::cout << "[WSI Registration] " << message << std::endl;
}

bool WSIRegistration::initializeCuda() {
    cudaInitialized_ = false;
#ifdef CUDA_AVAILABLE
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess && deviceCount > 0) {
        cudaInitialized_ = true;
        logProgress("CUDA initialized successfully with " + std::to_string(deviceCount) + " device(s)");
    } else {
        logProgress("CUDA initialization failed, using CPU implementation");
    }
#else
    logProgress("CUDA not available, using CPU implementation");
#endif
    return cudaInitialized_;
}

} // namespace wsi_registration