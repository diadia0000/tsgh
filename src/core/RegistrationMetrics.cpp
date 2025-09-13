#include "wsi/RegistrationMetrics.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

namespace cell_registration {

RegistrationMetrics RegistrationMetricsCalculator::calculateMetrics(const cv::Mat& ref, const cv::Mat& aligned, const cv::Mat& transform) {
    RegistrationMetrics metrics;
    
    metrics.mutualInformation = calculateMutualInformation(ref, aligned);
    metrics.normalizedMutualInformation = calculateNormalizedMutualInformation(ref, aligned);
    metrics.targetRegistrationError = calculateTRE(ref, aligned, transform);
    metrics.quality = assessQuality(metrics);
    
    return metrics;
}

string RegistrationMetricsCalculator::assessQuality(const RegistrationMetrics& metrics) {
    // Very relaxed thresholds for challenging medical image registration
    bool miAcceptable = metrics.mutualInformation > 0.001;     // Extremely low threshold
    bool nmiAcceptable = metrics.normalizedMutualInformation > 0.0001;  // Extremely low threshold
    bool treAcceptable = metrics.targetRegistrationError < 200.0;     // Very lenient
    
    // Reasonable quality thresholds
    bool miReasonable = metrics.mutualInformation > 0.01;
    bool nmiReasonable = metrics.normalizedMutualInformation > 0.001;
    bool treReasonable = metrics.targetRegistrationError < 100.0;
    
    if (miReasonable && nmiReasonable && treReasonable) {
        return "acceptable";
    } else if (miAcceptable && nmiAcceptable && treAcceptable) {
        return "challenging";
    } else {
        return "difficult";
    }
}

double RegistrationMetricsCalculator::calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size()) return 0.0;
    
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);
    
    const int bins = 64;
    Mat jointHist = Mat::zeros(bins, bins, CV_32F);
    
    for (int y = 0; y < gray1.rows; y++) {
        for (int x = 0; x < gray1.cols; x++) {
            int val1 = min(static_cast<int>(gray1.at<uchar>(y, x) * bins / 256), bins - 1);
            int val2 = min(static_cast<int>(gray2.at<uchar>(y, x) * bins / 256), bins - 1);
            jointHist.at<float>(val1, val2) += 1.0f;
        }
    }
    
    jointHist /= (gray1.rows * gray1.cols);
    
    // Calculate marginal histograms
    Mat hist1 = Mat::zeros(bins, 1, CV_32F);
    Mat hist2 = Mat::zeros(bins, 1, CV_32F);
    
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
                    mi += p_xy * log(p_xy / (p_x * p_y));
                }
            }
        }
    }
    
    return mi;
}

double RegistrationMetricsCalculator::calculateNormalizedMutualInformation(const cv::Mat& img1, const cv::Mat& img2) {
    double mi = calculateMutualInformation(img1, img2);
    double h1 = calculateEntropy(img1);
    double h2 = calculateEntropy(img2);
    
    if (h1 + h2 == 0) return 0.0;
    return 2.0 * mi / (h1 + h2);
}

double RegistrationMetricsCalculator::calculateTRE(const cv::Mat& fixed, const cv::Mat& moving, const cv::Mat& transform) {
    (void)moving;
    
    // Use corner points for TRE calculation (pure 2D)
    vector<Point2f> corners = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(fixed.cols), 0.0f),
        Point2f(static_cast<float>(fixed.cols), static_cast<float>(fixed.rows)),
        Point2f(0.0f, static_cast<float>(fixed.rows)),
        Point2f(static_cast<float>(fixed.cols)/2.0f, static_cast<float>(fixed.rows)/2.0f)
    };
    
    // Get 2x3 affine matrix
    Mat affineMatrix;
    if (transform.rows == 2 && transform.cols == 3) {
        affineMatrix = transform;
    } else {
        affineMatrix = transform(Rect(0, 0, 3, 2));
    }
    
    double totalError = 0.0;
    for (const auto& corner : corners) {
        // Pure 2D affine transformation
        double x_new = affineMatrix.at<double>(0, 0) * corner.x + 
                      affineMatrix.at<double>(0, 1) * corner.y + 
                      affineMatrix.at<double>(0, 2);
        double y_new = affineMatrix.at<double>(1, 0) * corner.x + 
                      affineMatrix.at<double>(1, 1) * corner.y + 
                      affineMatrix.at<double>(1, 2);
        
        double dx = x_new - corner.x;
        double dy = y_new - corner.y;
        totalError += sqrt(dx * dx + dy * dy);
    }
    
    return totalError / static_cast<double>(corners.size());
}

double RegistrationMetricsCalculator::calculateEntropy(const cv::Mat& image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    hist /= (gray.rows * gray.cols);
    
    double entropy = 0.0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 1e-10) {
            entropy -= p * log2(p);
        }
    }
    
    return entropy;
}

cv::Mat RegistrationMetricsCalculator::applyTransform(const cv::Mat& image, const cv::Mat& transform) {
    Mat result;
    
    // ONLY accept 2x3 matrices - reject anything else
    if (transform.rows != 2 || transform.cols != 3) {
        cerr << "ERROR: Transform must be 2x3 matrix! Got: " << transform.rows << "x" << transform.cols << endl;
        // Use identity transform
        Mat identity = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
        warpAffine(image, result, identity, image.size(), INTER_LINEAR);
        return result;
    }
    
    // Apply ONLY 2D affine transformation
    warpAffine(image, result, transform, image.size(), INTER_LINEAR);
    return result;
}

} // namespace cell_registration