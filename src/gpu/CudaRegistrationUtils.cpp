#include "wsi/CudaRegistration.h"
#include "wsi/CudaUtils.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace cuda_registration {

// CudaMemoryManager implementation
bool CudaMemoryManager::allocateGpuMemory(size_t bytes) {
    // MVP: Simulate GPU memory allocation using CPU memory
    try {
        gpuImageBuffer_ = cv::Mat::zeros(1, static_cast<int>(bytes), CV_8U);
        availableMemory_ = bytes;
        usedMemory_ = 0;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error allocating GPU memory: " << e.what() << std::endl;
        return false;
    }
}

void CudaMemoryManager::freeGpuMemory() {
    gpuImageBuffer_.release();
    availableMemory_ = 0;
    usedMemory_ = 0;
}

bool CudaMemoryManager::uploadImage(const cv::Mat& image) {
    size_t imageBytes = image.total() * image.elemSize();
    
    if (usedMemory_ + imageBytes > availableMemory_) {
        std::cerr << "Error: Insufficient GPU memory" << std::endl;
        return false;
    }
    
    // MVP: Simply store the image (in production, this would use cudaMemcpy)
    gpuImageBuffer_ = image.clone();
    usedMemory_ += imageBytes;
    
    return true;
}

cv::Mat CudaMemoryManager::downloadImage() {
    // MVP: Return the stored image (in production, this would use cudaMemcpy)
    return gpuImageBuffer_.clone();
}

// CudaSimilarityMetrics implementation
CudaSimilarityMetrics::CudaSimilarityMetrics(const CudaRegistrationConfig& config) 
    : config_(config) {
    memoryManager_ = std::make_unique<CudaMemoryManager>();
    memoryManager_->allocateGpuMemory(config_.maxGpuMemory);
}

double CudaSimilarityMetrics::calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
    // MVP: CPU implementation (in production, this would be a CUDA kernel)
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
    
    const int bins = 64;
    cv::Mat jointHist = cv::Mat::zeros(bins, bins, CV_32F);
    
    // Compute joint histogram (simulated GPU kernel)
    computeJointHistogramKernel(gray1, gray2, jointHist);
    
    // Normalize
    jointHist /= (gray1.rows * gray1.cols);
    
    // Calculate marginal histograms
    cv::Mat hist1 = cv::Mat::zeros(bins, 1, CV_32F);
    cv::Mat hist2 = cv::Mat::zeros(bins, 1, CV_32F);
    
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            hist1.at<float>(i) += jointHist.at<float>(i, j);
            hist2.at<float>(j) += jointHist.at<float>(i, j);
        }
    }
    
    // Calculate mutual information
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

double CudaSimilarityMetrics::calculateNormalizedCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
    // MVP: Use OpenCV's template matching (in production, custom CUDA kernel)
    cv::Mat result;
    cv::matchTemplate(img1, img2, result, cv::TM_CCOEFF_NORMED);
    
    double minVal, maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);
    return maxVal;
}

double CudaSimilarityMetrics::calculateSumOfSquaredDifferences(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size()) {
        return std::numeric_limits<double>::max();
    }
    
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_64F);
    
    cv::Scalar ssd = cv::sum(diff.mul(diff));
    return ssd[0];
}

std::vector<double> CudaSimilarityMetrics::calculateMIBatch(const cv::Mat& fixed, 
                                                          const std::vector<cv::Mat>& movingImages) {
    std::vector<double> results;
    results.reserve(movingImages.size());
    
    // MVP: Sequential processing (in production, parallel GPU processing)
    for (const auto& moving : movingImages) {
        double mi = calculateMutualInformation(fixed, moving);
        results.push_back(mi);
    }
    
    return results;
}

void CudaSimilarityMetrics::computeHistogramKernel(const cv::Mat& image, cv::Mat& histogram) {
    // MVP: CPU implementation of histogram computation
    const int bins = 64;
    histogram = cv::Mat::zeros(bins, 1, CV_32F);
    
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int bin = std::min(static_cast<int>(image.at<uchar>(y, x) * bins / 256), bins - 1);
            histogram.at<float>(bin) += 1.0f;
        }
    }
}

void CudaSimilarityMetrics::computeJointHistogramKernel(const cv::Mat& img1, const cv::Mat& img2, 
                                                       cv::Mat& jointHistogram) {
    // MVP: CPU implementation of joint histogram computation
    const int bins = 64;
    jointHistogram = cv::Mat::zeros(bins, bins, CV_32F);
    
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            int bin1 = std::min(static_cast<int>(img1.at<uchar>(y, x) * bins / 256), bins - 1);
            int bin2 = std::min(static_cast<int>(img2.at<uchar>(y, x) * bins / 256), bins - 1);
            jointHistogram.at<float>(bin1, bin2) += 1.0f;
        }
    }
}

// CudaImageProcessor implementation
CudaImageProcessor::CudaImageProcessor(const CudaRegistrationConfig& config) 
    : config_(config) {
    memoryManager_ = std::make_unique<CudaMemoryManager>();
    memoryManager_->allocateGpuMemory(config_.maxGpuMemory);
}

cv::Mat CudaImageProcessor::gaussianBlur(const cv::Mat& image, double sigma) {
    cv::Mat result;
    int kernelSize = static_cast<int>(2 * std::ceil(3 * sigma) + 1);
    cv::GaussianBlur(image, result, cv::Size(kernelSize, kernelSize), sigma);
    return result;
}

cv::Mat CudaImageProcessor::resize(const cv::Mat& image, cv::Size newSize) {
    cv::Mat result;
    cv::resize(image, result, newSize, 0, 0, cv::INTER_LINEAR);
    return result;
}

cv::Mat CudaImageProcessor::rotate(const cv::Mat& image, double angle) {
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cv::Mat result;
    cv::warpAffine(image, result, rotationMatrix, image.size(), 
                   cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return result;
}

cv::Mat CudaImageProcessor::warpAffine(const cv::Mat& image, const cv::Mat& transform) {
    cv::Mat result;
    cv::warpAffine(image, result, transform, image.size(), 
                   cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    return result;
}

std::vector<cv::Mat> CudaImageProcessor::buildGaussianPyramid(const cv::Mat& image, int levels) {
    std::vector<cv::Mat> pyramid;
    pyramid.reserve(levels);
    
    cv::Mat currentLevel = image.clone();
    pyramid.push_back(currentLevel);
    
    for (int i = 1; i < levels; ++i) {
        cv::Mat nextLevel;
        cv::pyrDown(currentLevel, nextLevel);
        pyramid.push_back(nextLevel);
        currentLevel = nextLevel;
        
        if (nextLevel.cols < 32 || nextLevel.rows < 32) {
            break;
        }
    }
    
    return pyramid;
}

std::vector<cv::Mat> CudaImageProcessor::processBatch(const std::vector<cv::Mat>& images,
                                                     std::function<cv::Mat(const cv::Mat&)> operation) {
    std::vector<cv::Mat> results;
    results.reserve(images.size());
    
    // MVP: Sequential processing (in production, parallel GPU processing)
    for (const auto& image : images) {
        results.push_back(operation(image));
    }
    
    return results;
}

} // namespace cuda_registration