#pragma once

#include "wsi/RegistrationMetrics.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include <chrono>

namespace cell_registration {

struct RegistrationResult {
    bool success = false;
    RegistrationMetrics metrics;
    cv::Mat transformMatrix;
    std::chrono::milliseconds processingTime{0};
    std::string errorMessage;
};

class CellRegistration {
public:
    CellRegistration();
    ~CellRegistration() = default;

    void setGpuEnabled(bool enabled) { gpuEnabled_ = enabled; }
    bool performRegistration(const std::string& inputDir, const std::string& outputDir);

private:
    bool gpuEnabled_ = false;
    
    // Images
    cv::Mat heImage_;    // Reference (HE staining)
    cv::Mat her2Image_;  // Her2 staining
    cv::Mat dishImage_;  // DISH staining
    
    // Multi-scale pyramid
    std::vector<double> pyramidScales_;
    
    // Results
    RegistrationResult her2Result_;
    RegistrationResult dishResult_;
    
    // Core methods
    bool loadImages(const std::string& inputDir);
    RegistrationResult registerToReference(const cv::Mat& reference, const cv::Mat& moving);
    bool saveResults(const std::string& outputDir);
    
    // Utility functions
    void logProgress(const std::string& message);
};

} // namespace cell_registration