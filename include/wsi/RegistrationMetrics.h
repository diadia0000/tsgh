#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cell_registration {

struct RegistrationMetrics {
    double mutualInformation = 0.0;
    double normalizedMutualInformation = 0.0;
    double targetRegistrationError = 0.0;
    std::string quality = "bad";  // good/normal/bad
};

class RegistrationMetricsCalculator {
public:
    RegistrationMetricsCalculator() = default;
    ~RegistrationMetricsCalculator() = default;

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
};

} // namespace cell_registration