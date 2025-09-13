#include "wsi/ImagePreprocessing.h"
#include <opencv2/opencv.hpp>

using namespace cv;

namespace cell_registration {

void ImagePreprocessing::preprocessMedicalImage(const cv::Mat& input, cv::Mat& output) {
    // Structure-focused preprocessing for medical images
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    
    // Morphological operations to enhance structures
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat opened, closed;
    morphologyEx(gray, opened, MORPH_OPEN, kernel);
    morphologyEx(opened, closed, MORPH_CLOSE, kernel);
    
    // Multi-scale edge detection
    Mat edges1, edges2, edges3;
    Canny(closed, edges1, 30, 90);   // Fine edges
    Canny(closed, edges2, 50, 150);  // Medium edges
    Canny(closed, edges3, 80, 240);  // Coarse edges
    
    // Combine multi-scale edges
    Mat combinedEdges;
    addWeighted(edges1, 0.5, edges2, 0.3, 0, combinedEdges);
    addWeighted(combinedEdges, 1.0, edges3, 0.2, 0, combinedEdges);
    
    // Structure enhancement with gradient
    Mat gradX, gradY, gradient;
    Sobel(closed, gradX, CV_32F, 1, 0, 3);
    Sobel(closed, gradY, CV_32F, 0, 1, 3);
    magnitude(gradX, gradY, gradient);
    gradient.convertTo(gradient, CV_8U);
    
    // Combine structure features
    Mat structural;
    addWeighted(combinedEdges, 0.6, gradient, 0.4, 0, structural);
    
    // Convert back to color for consistency
    cvtColor(structural, output, COLOR_GRAY2BGR);
}

void ImagePreprocessing::enhanceStructures(const cv::Mat& input, cv::Mat& output) {
    Mat gray;
    if (input.channels() == 3) {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Apply CLAHE for local contrast enhancement
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, output);
    
    if (input.channels() == 3) {
        cvtColor(output, output, COLOR_GRAY2BGR);
    }
}

void ImagePreprocessing::normalizeIntensity(const cv::Mat& input, cv::Mat& output) {
    Mat normalized;
    normalize(input, normalized, 0, 255, NORM_MINMAX);
    output = normalized;
}

} // namespace cell_registration