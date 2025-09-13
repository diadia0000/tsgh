#include "wsi/CudaUtils.h"
#include "wsi/CudaRegistration.h"
#include <iostream>
#include <cmath>

namespace cuda_registration {

// Static initialization
bool CudaUtils::cudaInitialized_ = false;

// CudaUtils implementation
std::vector<CudaDeviceInfo> CudaUtils::getAvailableDevices() {
    std::vector<CudaDeviceInfo> devices;
    
    // MVP: Return simulated device info
    CudaDeviceInfo device;
    device.deviceId = 0;
    device.name = "Simulated CUDA Device";
    device.totalMemory = static_cast<size_t>(8) * 1024 * 1024 * 1024; // 8GB
    device.freeMemory = static_cast<size_t>(6) * 1024 * 1024 * 1024;  // 6GB
    device.computeCapability = 75; // Simulate Turing architecture
    device.isAvailable = true;
    
    devices.push_back(device);
    
    return devices;
}

bool CudaUtils::isDeviceAvailable(int deviceId) {
    // MVP: Always return true for device 0
    return deviceId == 0;
}

CudaDeviceInfo CudaUtils::getDeviceInfo(int deviceId) {
    auto devices = getAvailableDevices();
    if (deviceId >= 0 && deviceId < static_cast<int>(devices.size())) {
        return devices[deviceId];
    }
    
    return CudaDeviceInfo{}; // Return empty info for invalid device
}

void CudaUtils::warmupGpu(int deviceId) {
    // MVP: No-op for simulation
    std::cout << "GPU warmup completed for device " << deviceId << std::endl;
}

double CudaUtils::benchmarkDevice(int /* deviceId */) {
    // MVP: Return simulated benchmark score
    return 1000.0; // GFLOPS
}

size_t CudaUtils::getOptimalTileSize(int /* deviceId */, const cv::Size& imageSize) {
    // MVP: Return reasonable tile size based on image dimensions
    int tileSize = std::min(512, std::min(imageSize.width, imageSize.height) / 4);
    return static_cast<size_t>(tileSize);
}

bool CudaUtils::checkMemoryRequirements(int deviceId, size_t requiredBytes) {
    CudaDeviceInfo info = getDeviceInfo(deviceId);
    return info.freeMemory >= requiredBytes;
}

void CudaUtils::initializeCuda() {
    if (!cudaInitialized_) {
        // MVP: No actual CUDA initialization needed
        cudaInitialized_ = true;
    }
}

// CudaRegistrationFactory implementation
std::unique_ptr<CudaRegistrationEngine> CudaRegistrationFactory::createEngine(
    const CudaRegistrationConfig& config) {
    return std::make_unique<CudaRegistrationEngine>(config);
}

std::unique_ptr<CudaSimilarityMetrics> CudaRegistrationFactory::createSimilarityMetrics(
    const CudaRegistrationConfig& config) {
    return std::make_unique<CudaSimilarityMetrics>(config);
}

std::unique_ptr<CudaImageProcessor> CudaRegistrationFactory::createImageProcessor(
    const CudaRegistrationConfig& config) {
    return std::make_unique<CudaImageProcessor>(config);
}

CudaRegistrationConfig CudaRegistrationFactory::getOptimalConfig(int deviceId) {
    CudaRegistrationConfig config;
    
    CudaDeviceInfo info = CudaUtils::getDeviceInfo(deviceId);
    if (info.isAvailable) {
        config.deviceId = deviceId;
        config.maxGpuMemory = static_cast<size_t>(info.freeMemory * 0.8); // Use 80% of available memory
        config.blockSize = (info.computeCapability >= 70) ? 32 : 16;
        config.useSharedMemory = true;
    }
    
    return config;
}

} // namespace cuda_registration