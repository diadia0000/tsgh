#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

// Note: This is a header-only CUDA interface for MVP
// Full CUDA implementation would require .cu files and nvcc compilation

namespace cuda_registration {

struct CudaDeviceInfo {
    int deviceId = -1;
    std::string name;
    size_t totalMemory = 0;
    size_t freeMemory = 0;
    int computeCapability = 0;
    bool isAvailable = false;
};

struct CudaRegistrationConfig {
    int deviceId = 0;
    size_t maxGpuMemory = 1024 * 1024 * 1024; // 1GB default
    int blockSize = 16;
    bool useSharedMemory = true;
    bool enableProfiling = false;
};

class CudaMemoryManager {
public:
    CudaMemoryManager() = default;
    ~CudaMemoryManager() = default;

    // Memory allocation (CPU-side interface for MVP)
    bool allocateGpuMemory(size_t bytes);
    void freeGpuMemory();
    
    // Data transfer simulation
    bool uploadImage(const cv::Mat& image);
    cv::Mat downloadImage();
    
    // Memory info
    size_t getAvailableMemory() const { return availableMemory_; }
    size_t getUsedMemory() const { return usedMemory_; }

private:
    size_t availableMemory_ = 0;
    size_t usedMemory_ = 0;
    cv::Mat gpuImageBuffer_; // CPU simulation of GPU memory
};

class CudaSimilarityMetrics {
public:
    explicit CudaSimilarityMetrics(const CudaRegistrationConfig& config);
    ~CudaSimilarityMetrics() = default;

    // GPU-accelerated similarity metrics (CPU implementation for MVP)
    double calculateMutualInformation(const cv::Mat& img1, const cv::Mat& img2);
    double calculateNormalizedCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2);
    double calculateSumOfSquaredDifferences(const cv::Mat& img1, const cv::Mat& img2);
    
    // Batch processing
    std::vector<double> calculateMIBatch(const cv::Mat& fixed, 
                                        const std::vector<cv::Mat>& movingImages);

private:
    CudaRegistrationConfig config_;
    std::unique_ptr<CudaMemoryManager> memoryManager_;
    
    // Internal GPU kernels (CPU simulation)
    void computeHistogramKernel(const cv::Mat& image, cv::Mat& histogram);
    void computeJointHistogramKernel(const cv::Mat& img1, const cv::Mat& img2, 
                                   cv::Mat& jointHistogram);
};

class CudaImageProcessor {
public:
    explicit CudaImageProcessor(const CudaRegistrationConfig& config);
    ~CudaImageProcessor() = default;

    // GPU-accelerated image operations
    cv::Mat gaussianBlur(const cv::Mat& image, double sigma);
    cv::Mat resize(const cv::Mat& image, cv::Size newSize);
    cv::Mat rotate(const cv::Mat& image, double angle);
    cv::Mat warpAffine(const cv::Mat& image, const cv::Mat& transform);
    
    // Multi-scale processing
    std::vector<cv::Mat> buildGaussianPyramid(const cv::Mat& image, int levels);
    
    // Batch operations
    std::vector<cv::Mat> processBatch(const std::vector<cv::Mat>& images,
                                     std::function<cv::Mat(const cv::Mat&)> operation);

private:
    CudaRegistrationConfig config_;
    std::unique_ptr<CudaMemoryManager> memoryManager_;
};

class CudaRegistrationEngine {
public:
    explicit CudaRegistrationEngine(const CudaRegistrationConfig& config);
    ~CudaRegistrationEngine() = default;

    // Main registration interface
    bool initialize();
    void shutdown();
    
    // GPU-accelerated registration
    cv::Mat registerImages(const cv::Mat& fixed, const cv::Mat& moving);
    
    // Multi-resolution registration
    cv::Mat multiResolutionRegistration(const cv::Mat& fixed, const cv::Mat& moving,
                                       const std::vector<int>& pyramidLevels);
    
    // Batch registration
    std::vector<cv::Mat> registerImageBatch(const cv::Mat& fixed,
                                           const std::vector<cv::Mat>& movingImages);
    
    // Performance monitoring
    double getLastRegistrationTime() const { return lastRegistrationTime_; }
    size_t getGpuMemoryUsage() const;

private:
    CudaRegistrationConfig config_;
    std::unique_ptr<CudaSimilarityMetrics> similarityMetrics_;
    std::unique_ptr<CudaImageProcessor> imageProcessor_;
    
    bool isInitialized_ = false;
    double lastRegistrationTime_ = 0.0;
    
    // Internal registration methods
    cv::Mat optimizeTransform(const cv::Mat& fixed, const cv::Mat& moving);
    cv::Mat gradientDescentOptimization(const cv::Mat& fixed, const cv::Mat& moving);
};

// Utility functions
class CudaUtils {
public:
    // Device management
    static std::vector<CudaDeviceInfo> getAvailableDevices();
    static bool isDeviceAvailable(int deviceId);
    static CudaDeviceInfo getDeviceInfo(int deviceId);
    
    // Performance utilities
    static void warmupGpu(int deviceId);
    static double benchmarkDevice(int deviceId);
    
    // Memory utilities
    static size_t getOptimalTileSize(int deviceId, const cv::Size& imageSize);
    static bool checkMemoryRequirements(int deviceId, size_t requiredBytes);

private:
    static bool cudaInitialized_;
    static void initializeCuda();
};

// Factory class for creating CUDA-accelerated components
class CudaRegistrationFactory {
public:
    static std::unique_ptr<CudaRegistrationEngine> createEngine(
        const CudaRegistrationConfig& config = CudaRegistrationConfig{});
    
    static std::unique_ptr<CudaSimilarityMetrics> createSimilarityMetrics(
        const CudaRegistrationConfig& config = CudaRegistrationConfig{});
    
    static std::unique_ptr<CudaImageProcessor> createImageProcessor(
        const CudaRegistrationConfig& config = CudaRegistrationConfig{});
    
    static CudaRegistrationConfig getOptimalConfig(int deviceId = 0);
};

} // namespace cuda_registration