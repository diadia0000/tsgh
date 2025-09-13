#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include <iostream>

namespace cell_registration {

struct RegistrationMetrics;
class RegistrationMetricsCalculator;

class RegistrationStages {
public:
    RegistrationStages() = default;
    ~RegistrationStages() = default;

    // Stage implementations
    cv::Mat featureBasedAlignment(const cv::Mat& ref, const cv::Mat& mov);
    cv::Mat mutualInfoAlignment(const cv::Mat& ref, const cv::Mat& mov, const cv::Mat& initial);
    cv::Mat bsplineAlignment(const cv::Mat& ref, const cv::Mat& mov, const cv::Mat& initial);

    // Transform estimation
    cv::Mat estimateRobustTransform(const std::vector<cv::Point2f>& srcPoints,
                                   const std::vector<cv::Point2f>& dstPoints);

    // Optimization methods
    cv::Mat performMultiParameterOptimization(const cv::Mat& ref, const cv::Mat& mov,
                                             cv::Mat& bestTransform, double& bestMI);
    cv::Mat performFineTuningOptimization(const cv::Mat& ref, const cv::Mat& mov,
                                         cv::Mat& bestTransform, double& bestMI);

    // Fine adjustment methods
    bool optimizeTranslation(const cv::Mat& ref, const cv::Mat& mov,
                           cv::Mat& bestTransform, double& bestMI,
                           double tStep, RegistrationMetricsCalculator& metrics);
    bool optimizeRotation(const cv::Mat& ref, const cv::Mat& mov,
                        cv::Mat& bestTransform, double& bestMI,
                        double rStep, RegistrationMetricsCalculator& metrics);
    bool optimizeScale(const cv::Mat& ref, const cv::Mat& mov,
                     cv::Mat& bestTransform, double& bestMI,
                     double sStep, RegistrationMetricsCalculator& metrics);

    bool performFineTranslationAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                        cv::Mat& bestTransform, double& bestMI,
                                        double tStep, RegistrationMetricsCalculator& metrics);
    bool performFineRotationAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                     cv::Mat& bestTransform, double& bestMI,
                                     double rStep, RegistrationMetricsCalculator& metrics);
    bool performFineScaleAdjustment(const cv::Mat& ref, const cv::Mat& mov,
                                  cv::Mat& bestTransform, double& bestMI,
                                  double sStep, RegistrationMetricsCalculator& metrics);

    // Validation and logging
    void validateAndLogTransform(const cv::Mat& transform, 
                               const std::vector<cv::Point2f>& srcPoints,
                               const std::vector<cv::Point2f>& dstPoints);
    void logFinalOptimizationResult(const cv::Mat& bestTransform, double bestMI);
    void logFinalBSplineResult(const cv::Mat& bestTransform, double bestMI);

private:
    void logProgress(const std::string& message);
};

} // namespace cell_registration