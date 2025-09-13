#pragma once

#include <opencv2/opencv.hpp>

namespace cell_registration {

class ImagePreprocessing {
public:
    ImagePreprocessing() = default;
    ~ImagePreprocessing() = default;

    // Image preprocessing methods
    void preprocessMedicalImage(const cv::Mat& input, cv::Mat& output);
    
private:
    // Internal preprocessing utilities
    void enhanceStructures(const cv::Mat& input, cv::Mat& output);
    void normalizeIntensity(const cv::Mat& input, cv::Mat& output);
};

} // namespace cell_registration