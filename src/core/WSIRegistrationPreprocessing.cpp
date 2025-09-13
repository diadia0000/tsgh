#include "wsi/WSIRegistration.h"
#include <algorithm>

namespace wsi_registration {

cv::Mat WSIRegistration::preprocessImage(const cv::Mat& image, const std::string& imageType) const {
    cv::Mat processed;
    
    if (imageType == "fluorescence" || imageType == "fish") {
        // 螢光影像預處理
        if (image.type() != CV_8UC3) {
            image.convertTo(processed, CV_8UC3);
        } else {
            processed = image.clone();
        }
        
        // 檢查是否為暗螢光影像並反轉
        cv::Scalar meanIntensity = cv::mean(processed);
        bool isDark = (meanIntensity[0] + meanIntensity[1] + meanIntensity[2]) / 3.0 < 50;
        
        if (isDark) {
            cv::bitwise_not(processed, processed);
            logProgress("反轉暗螢光影像以改善處理效果");
        }
        
        // 螢光影像對比度增強
        cv::Mat lab;
        cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labChannels;
        cv::split(lab, labChannels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);
        
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
        
        // 螢光影像去噪
        cv::Mat denoised;
        cv::fastNlMeansDenoisingColored(processed, denoised, 10, 10, 7, 21);
        processed = denoised;
        
    } else {
        // 明場影像預處理 (H&E, HER2)
        if (image.type() != CV_8UC3) {
            image.convertTo(processed, CV_8UC3);
        } else {
            processed = image.clone();
        }
        
        // 明場影像色彩正規化
        processed = normalizeStaining(processed);
        
        // 明場影像對比度增強
        cv::Mat lab;
        cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labChannels;
        cv::split(lab, labChannels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);
        
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, processed, cv::COLOR_Lab2BGR);
        
        // 明場影像銳化
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
                         0, -1, 0,
                         -1, 5, -1,
                         0, -1, 0);
        cv::Mat sharpened;
        cv::filter2D(processed, sharpened, -1, kernel);
        processed = sharpened;
    }
    
    return processed;
}

cv::Mat WSIRegistration::normalizeStaining(const cv::Mat& image) const {
    // 簡化的染色正規化實現
    // 實際應用中可能需要更複雜的色彩去卷積方法
    
    cv::Mat normalized;
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    
    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);
    
    // 正規化 L 通道
    cv::Mat& lChannel = labChannels[0];
    cv::Scalar mean, stddev;
    cv::meanStdDev(lChannel, mean, stddev);
    
    // 目標統計值 (經驗值)
    double targetMean = 128.0;
    double targetStd = 30.0;
    
    lChannel.convertTo(lChannel, CV_32F);
    lChannel = (lChannel - mean[0]) * (targetStd / stddev[0]) + targetMean;
    lChannel.convertTo(lChannel, CV_8U);
    
    // 限制值範圍
    cv::threshold(lChannel, lChannel, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(lChannel, lChannel, 0, 0, cv::THRESH_TOZERO);
    
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, normalized, cv::COLOR_Lab2BGR);
    
    return normalized;
}

} // namespace wsi_registration