#pragma once

#include "WSIRegistration.h"

namespace wsi_registration {

class WSIRegistrationStages {
public:
    explicit WSIRegistrationStages(const RegistrationParams& params);
    ~WSIRegistrationStages() = default;

    // Four-stage registration workflow
    RegistrationResult performFourStageRegistration(const cv::Mat& reference, 
                                                   const cv::Mat& moving,
                                                   const std::string& movingType);

private:
    RegistrationParams params_;
    
    // Stage implementations
    StageResult performFeatureBasedCoarseAlignment(const cv::Mat& reference, 
                                                  const cv::Mat& moving,
                                                  const std::string& movingType);
    
    StageResult performMutualInfoAffineAlignment(const cv::Mat& reference, 
                                                const cv::Mat& moving,
                                                const cv::Mat& initialTransform);
    
    StageResult performBSplineNonRigidAlignment(const cv::Mat& reference, 
                                               const cv::Mat& moving,
                                               const cv::Mat& initialTransform);
    
    // Helper methods
    std::vector<cv::DMatch> matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2) const;
    cv::Mat estimateTransformRANSAC(const std::vector<cv::Point2f>& srcPoints,
                                   const std::vector<cv::Point2f>& dstPoints) const;
    
    void logProgress(const std::string& message) const;
};

} // namespace wsi_registration