#include "wsi/WSIRegistration.h"
#include <cmath>
#include <algorithm>

namespace wsi_registration {

double WSIRegistration::calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
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
    
    const int bins = params_.histogramBins;
    
    // 計算聯合直方圖
    cv::Mat jointHist = cv::Mat::zeros(bins, bins, CV_32F);
    
    for (int y = 0; y < gray1.rows; y++) {
        for (int x = 0; x < gray1.cols; x++) {
            int val1 = std::min(static_cast<int>(gray1.at<uchar>(y, x) * bins / 256), bins - 1);
            int val2 = std::min(static_cast<int>(gray2.at<uchar>(y, x) * bins / 256), bins - 1);
            jointHist.at<float>(val1, val2) += 1.0f;
        }
    }
    
    jointHist /= (gray1.rows * gray1.cols);
    
    // 計算邊際直方圖
    cv::Mat hist1 = cv::Mat::zeros(bins, 1, CV_32F);
    cv::Mat hist2 = cv::Mat::zeros(bins, 1, CV_32F);
    
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            hist1.at<float>(i) += jointHist.at<float>(i, j);
            hist2.at<float>(j) += jointHist.at<float>(i, j);
        }
    }
    
    // 計算互信息
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

double WSIRegistration::calculateNormalizedMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const {
    double mi = calculateMutualInformation(img1, img2);
    double h1 = calculateEntropy(img1);
    double h2 = calculateEntropy(img2);
    
    if (h1 + h2 == 0) return 0.0;
    
    return 2.0 * mi / (h1 + h2);
}

double WSIRegistration::calculateTargetRegistrationError(const cv::Mat& fixed, const cv::Mat& moving, 
                                                        const cv::Mat& transform) const {
    // 使用角點進行簡化的 TRE 計算
    std::vector<cv::Point2f> corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(fixed.cols), 0.0f),
        cv::Point2f(static_cast<float>(fixed.cols), static_cast<float>(fixed.rows)),
        cv::Point2f(0.0f, static_cast<float>(fixed.rows)),
        cv::Point2f(static_cast<float>(fixed.cols/2), static_cast<float>(fixed.rows/2))  // 中心點
    };
    
    double totalError = 0.0;
    
    for (const auto& corner : corners) {
        // 應用變換到角點
        cv::Mat point = (cv::Mat_<double>(3, 1) << corner.x, corner.y, 1);
        cv::Mat transformedPoint = transform * point;
        
        // 計算歐幾里得距離
        double dx = transformedPoint.at<double>(0, 0) - corner.x;
        double dy = transformedPoint.at<double>(1, 0) - corner.y;
        totalError += std::sqrt(dx * dx + dy * dy);
    }
    
    return totalError / corners.size();
}

double WSIRegistration::calculateEntropy(const cv::Mat& image) const {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 計算直方圖
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // 正規化
    hist /= (gray.rows * gray.cols);
    
    // 計算熵
    double entropy = 0.0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 1e-10) {
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

} // namespace wsi_registration