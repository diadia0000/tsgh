#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include <chrono>

namespace cell_registration {

struct RegistrationMetrics {
    double mutualInformation = 0.0;
    double normalizedMutualInformation = 0.0;
    double targetRegistrationError = 0.0;
    std::string quality = "bad";  // good/normal/bad
};

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
    
    // Registration stages
    cv::Mat featureBasedAlignment(const cv::Mat& ref, const cv::Mat& mov);
    cv::Mat mutualInfoAlignment(const cv::Mat& ref, const cv::Mat& mov, const cv::Mat& initial);
    cv::Mat bsplineAlignment(const cv::Mat& ref, const cv::Mat& mov, const cv::Mat& initial);
    
    // Quality assessment
    RegistrationMetrics calculateMetrics(const cv::Mat& ref, const cv::Mat& aligned, const cv::Mat& transform);
    std::string assessQuality(const RegistrationMetrics& metrics);
    
    // Metric calculation methods
    double calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2);
    double calculateNormalizedMutualInformation(const cv::Mat& img1, const cv::Mat& img2);
    double calculateTRE(const cv::Mat& fixed, const cv::Mat& moving, const cv::Mat& transform);
    double calculateEntropy(const cv::Mat& image);
    
    // Utility functions
    cv::Mat applyTransform(const cv::Mat& image, const cv::Mat& transform);
    void preprocessMedicalImage(const cv::Mat& input, cv::Mat& output);
    void logProgress(const std::string& message);
};

} // namespace cell_registration