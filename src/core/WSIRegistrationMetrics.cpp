#include "wsi/WSIRegistration.h"
#include <cmath>
#include <algorithm>

namespace wsi_registration {

double WSIRegistration::calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
    cv::Mat gray1, gray2;
    
    // 改進的灰階轉換 - 針對不同染色優化
    if (img1.channels() == 3) {
        // 使用加權灰階轉換，保留更多結構信息
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        // 增強對比度
        cv::equalizeHist(gray1, gray1);
    } else {
        gray1 = img1.clone();
    }
    
    if (img2.channels() == 3) {
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray2, gray2);
    } else {
        gray2 = img2.clone();
    }
    
    // 增加直方圖bins數量以提高精度
    const int bins = std::max(params_.histogramBins, 128);
    
    // 使用浮點數精度計算聯合直方圖
    cv::Mat jointHist = cv::Mat::zeros(bins, bins, CV_64F);
    
    // 改進的直方圖計算 - 使用雙線性插值
    for (int y = 0; y < gray1.rows; y++) {
        for (int x = 0; x < gray1.cols; x++) {
            double val1 = static_cast<double>(gray1.at<uchar>(y, x)) * (bins - 1) / 255.0;
            double val2 = static_cast<double>(gray2.at<uchar>(y, x)) * (bins - 1) / 255.0;
            
            int i1 = static_cast<int>(val1);
            int j1 = static_cast<int>(val2);
            int i2 = std::min(i1 + 1, bins - 1);
            int j2 = std::min(j1 + 1, bins - 1);
            
            double di = val1 - i1;
            double dj = val2 - j1;
            
            // 雙線性插值更新直方圖
            jointHist.at<double>(i1, j1) += (1.0 - di) * (1.0 - dj);
            jointHist.at<double>(i1, j2) += (1.0 - di) * dj;
            jointHist.at<double>(i2, j1) += di * (1.0 - dj);
            jointHist.at<double>(i2, j2) += di * dj;
        }
    }
    
    // 正規化
    double totalPixels = gray1.rows * gray1.cols;
    jointHist /= totalPixels;
    
    // 計算邊際直方圖
    cv::Mat hist1 = cv::Mat::zeros(bins, 1, CV_64F);
    cv::Mat hist2 = cv::Mat::zeros(bins, 1, CV_64F);
    
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            hist1.at<double>(i) += jointHist.at<double>(i, j);
            hist2.at<double>(j) += jointHist.at<double>(i, j);
        }
    }
    
    // 計算互信息 - 使用更穩定的數值計算
    double mi = 0.0;
    const double epsilon = 1e-12;
    
    for (int i = 0; i < bins; i++) {
        for (int j = 0; j < bins; j++) {
            double p_xy = jointHist.at<double>(i, j);
            if (p_xy > epsilon) {
                double p_x = hist1.at<double>(i);
                double p_y = hist2.at<double>(j);
                if (p_x > epsilon && p_y > epsilon) {
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