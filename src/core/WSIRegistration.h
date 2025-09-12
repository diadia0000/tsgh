#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace wsi_registration {

// 配準階段枚舉
enum class RegistrationStage {
    FEATURE_BASED_COARSE,    // 特徵點粗對齊 (SIFT/ORB + RANSAC)
    MUTUAL_INFO_AFFINE,      // 互信息 + 仿射變換精準對齊
    BSPLINE_NONRIGID,        // B-spline FFD 非剛體對齊
    COMPLETE
};

struct RegistrationParams {
    // 基本參數
    std::string referenceType = "HE";  // 基準影像類型
    bool enableCudaAcceleration = true;
    
    // 階段1: 特徵點粗對齊
    std::string featureDetector = "SIFT";  // SIFT 或 ORB
    int maxFeatures = 5000;
    double ransacThreshold = 3.0;
    int minMatchCount = 50;
    double ratioTestThreshold = 0.7;
    
    // 階段2: 互信息精準對齊
    std::vector<int> pyramidLevels = {4, 2, 1};
    int maxIterationsAffine = 500;
    double convergenceThreshold = 1e-6;
    int histogramBins = 64;
    
    // 階段3: B-spline 非剛體對齊
    std::vector<int> gridSpacing = {32, 16, 8};
    int maxIterationsBSpline = 200;
    double bsplineRegularization = 0.01;
    
    // 品質控制
    bool enablePixelLevelAlignment = true;
    double targetTRE = 2.0;  // 目標配準誤差 (像素)
    
    // WSI 特定參數
    int tileSize = 1024;
    int overlapSize = 128;
    double downsampleFactor = 4.0;  // 初始降採樣倍率
};

// 階段結果結構
struct StageResult {
    bool success = false;
    RegistrationStage stage;
    std::chrono::milliseconds processingTime{0};
    cv::Mat transformMatrix;
    std::string errorMessage;
    
    // 品質指標
    double mutualInformation = 0.0;
    double normalizedMutualInformation = 0.0;
    double targetRegistrationError = 0.0;
    int featureMatches = 0;  // 特徵點匹配數量
};

struct RegistrationResult {
    bool success = false;
    std::vector<StageResult> stageResults;
    std::chrono::milliseconds totalProcessingTime{0};
    cv::Mat finalTransformMatrix;
    std::string errorMessage;
    
    // 最終品質指標
    double finalMI = 0.0;
    double finalNMI = 0.0;
    double finalTRE = 0.0;
    
    // CUDA 使用統計
    bool usedCuda = false;
    size_t gpuMemoryUsed = 0;
};

class WSIRegistration {
public:
    WSIRegistration();
    explicit WSIRegistration(const RegistrationParams& params);
    ~WSIRegistration() = default;

    // 主要工作流程函數
    bool loadWSI(const std::string& her2_path, 
                 const std::string& he_path, 
                 const std::string& fish_path);

    // 四階段配準工作流程
    bool performRegistration();
    
    bool saveAlignedImages(const std::string& output_dir);
    
    // 配置
    void setParams(const RegistrationParams& params) { params_ = params; }
    const RegistrationParams& getParams() const { return params_; }
    
    // 結果存取
    const RegistrationResult& getHER2Result() const { return her2Result_; }
    const RegistrationResult& getFISHResult() const { return fishResult_; }
    
    // 品質評估指標
    double calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const;
    double calculateNormalizedMutualInformation(const cv::Mat& img1, const cv::Mat& img2) const;
    double calculateTargetRegistrationError(const cv::Mat& fixed, const cv::Mat& moving, 
                                          const cv::Mat& transform) const;
    
    bool generateReport(const std::string& report_path) const;

private:
    RegistrationParams params_;
    
    // 輸入影像
    cv::Mat her2Image_;
    cv::Mat heImage_;      // 基準影像
    cv::Mat fishImage_;
    
    // 配準結果
    RegistrationResult her2Result_;
    RegistrationResult fishResult_;
    
    // 對齊後影像
    cv::Mat alignedHER2_;
    cv::Mat alignedFISH_;
    
    // CUDA 相關
    bool cudaInitialized_ = false;
    int cudaDeviceId_ = 0;
    
    // 內部方法
    bool validateInputs() const;
    bool initializeCuda();
    
    // 四階段配準工作流程
    RegistrationResult performFourStageRegistration(const cv::Mat& reference, 
                                                   const cv::Mat& moving,
                                                   const std::string& movingType);
    
    // 階段1: 特徵點粗對齊
    StageResult performFeatureBasedCoarseAlignment(const cv::Mat& reference, 
                                                  const cv::Mat& moving,
                                                  const std::string& movingType);
    
    // 階段2: 互信息精準對齊
    StageResult performMutualInfoAffineAlignment(const cv::Mat& reference, 
                                                const cv::Mat& moving,
                                                const cv::Mat& initialTransform);
    
    // 階段3: B-spline 非剛體對齊
    StageResult performBSplineNonRigidAlignment(const cv::Mat& reference, 
                                               const cv::Mat& moving,
                                               const cv::Mat& initialTransform);
    
    // 特徵檢測與匹配
    std::vector<cv::KeyPoint> detectFeatures(const cv::Mat& image, 
                                            const std::string& detectorType) const;
    std::vector<cv::DMatch> matchFeatures(const cv::Mat& desc1, 
                                         const cv::Mat& desc2) const;
    cv::Mat estimateTransformRANSAC(const std::vector<cv::Point2f>& srcPoints,
                                   const std::vector<cv::Point2f>& dstPoints) const;
    
    // 互信息優化
    cv::Mat optimizeAffineTransformMI(const cv::Mat& reference, 
                                     const cv::Mat& moving,
                                     const cv::Mat& initialTransform) const;
    
    // B-spline 變形場
    cv::Mat generateBSplineDeformationField(const cv::Mat& reference,
                                           const cv::Mat& moving,
                                           const cv::Mat& initialTransform,
                                           int gridSpacing) const;
    
    // 影像預處理
    cv::Mat preprocessImage(const cv::Mat& image, const std::string& imageType) const;
    cv::Mat preprocessWSITile(const cv::Mat& tile, const std::string& imageType) const;
    
    // 變換應用
    cv::Mat applyAffineTransform(const cv::Mat& image, const cv::Mat& transform) const;
    cv::Mat applyBSplineTransform(const cv::Mat& image, const cv::Mat& deformationField) const;
    
    // CUDA 加速函數
    cv::Mat calculateMutualInformationCuda(const cv::Mat& img1, const cv::Mat& img2) const;
    cv::Mat optimizeTransformCuda(const cv::Mat& reference, const cv::Mat& moving,
                                 const cv::Mat& initialTransform) const;
    
    // 品質評估
    double calculateEntropy(const cv::Mat& image) const;
    double calculateJointEntropy(const cv::Mat& img1, const cv::Mat& img2) const;
    
    // 工具函數
    void logProgress(const std::string& message) const;
    std::string getCurrentTimestamp() const;
    cv::Mat createQualityOverlay(const cv::Mat& reference, const cv::Mat& aligned) const;
};

} // namespace wsi_registration