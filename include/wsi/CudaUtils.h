#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

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
    static std::unique_ptr<class CudaRegistrationEngine> createEngine(
        const CudaRegistrationConfig& config = CudaRegistrationConfig{});
    
    static std::unique_ptr<class CudaSimilarityMetrics> createSimilarityMetrics(
        const CudaRegistrationConfig& config = CudaRegistrationConfig{});
    
    static std::unique_ptr<class CudaImageProcessor> createImageProcessor(
        const CudaRegistrationConfig& config = CudaRegistrationConfig{});
    
    static CudaRegistrationConfig getOptimalConfig(int deviceId = 0);
};

} // namespace cuda_registration