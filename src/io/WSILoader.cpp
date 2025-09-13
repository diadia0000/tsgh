#include "wsi/WSILoader.h"
#include "wsi/WSILoaderExtensions.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace wsi_io {

// WSILoader static methods
WSIFormat WSILoader::detectFormat(const std::string& filepath) {
    std::filesystem::path path(filepath);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".svs") {
        return WSIFormat::SVS;
    } else if (extension == ".tiff" || extension == ".tif") {
        return WSIFormat::TIFF;
    } else if (extension == ".czi") {
        return WSIFormat::CZI;
    } else if (extension == ".ndpi") {
        return WSIFormat::NDPI;
    }
    
    return WSIFormat::UNKNOWN;
}

std::unique_ptr<WSILoader> WSILoader::createLoader(const std::string& filepath) {
    WSIFormat format = detectFormat(filepath);
    
    switch (format) {
        case WSIFormat::CZI:
            return std::make_unique<CZILoader>();
        case WSIFormat::TIFF:
        case WSIFormat::SVS:
        default:
            return std::make_unique<OpenCVWSILoader>();
    }
}

// OpenCVWSILoader implementation
bool OpenCVWSILoader::open(const std::string& filepath) {
    filepath_ = filepath;
    
    // Load the full image
    fullImage_ = cv::imread(filepath, cv::IMREAD_COLOR);
    
    if (fullImage_.empty()) {
        std::cerr << "Error: Cannot load image from " << filepath << std::endl;
        return false;
    }
    
    // Initialize metadata
    metadata_.width = fullImage_.cols;
    metadata_.height = fullImage_.rows;
    metadata_.format = detectFormat(filepath);
    metadata_.vendor = "OpenCV";
    
    // Build multi-resolution pyramid
    buildPyramid();
    
    metadata_.levels = static_cast<int>(pyramidLevels_.size());
    
    // Calculate level dimensions and downsamples
    metadata_.levelDimensions.clear();
    metadata_.levelDownsamples.clear();
    
    for (int i = 0; i < metadata_.levels; ++i) {
        metadata_.levelDimensions.push_back(cv::Size(pyramidLevels_[i].cols, pyramidLevels_[i].rows));
        metadata_.levelDownsamples.push_back(std::pow(2.0, i));
    }
    
    isOpen_ = true;
    return true;
}

void OpenCVWSILoader::close() {
    fullImage_.release();
    pyramidLevels_.clear();
    isOpen_ = false;
}

cv::Mat OpenCVWSILoader::readRegion(int x, int y, int width, int height, int level) {
    if (!isOpen_ || level >= static_cast<int>(pyramidLevels_.size())) {
        return cv::Mat();
    }
    
    const cv::Mat& levelImage = pyramidLevels_[level];
    return extractRegion(levelImage, x, y, width, height);
}

cv::Mat OpenCVWSILoader::readTile(const TileInfo& tile) {
    return readRegion(tile.x, tile.y, tile.width, tile.height, tile.level);
}

cv::Mat OpenCVWSILoader::getThumbnail(int maxSize) {
    if (!isOpen_) {
        return cv::Mat();
    }
    
    // Find the best level for thumbnail
    int bestLevel = 0;
    for (int i = 0; i < metadata_.levels; ++i) {
        cv::Size levelSize = metadata_.levelDimensions[i];
        if (std::max(levelSize.width, levelSize.height) <= maxSize) {
            bestLevel = i;
            break;
        }
    }
    
    cv::Mat thumbnail = pyramidLevels_[bestLevel].clone();
    
    // Further resize if needed
    if (std::max(thumbnail.cols, thumbnail.rows) > maxSize) {
        double scale = static_cast<double>(maxSize) / std::max(thumbnail.cols, thumbnail.rows);
        cv::resize(thumbnail, thumbnail, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    
    return thumbnail;
}

cv::Size OpenCVWSILoader::getLevelDimensions(int level) const {
    if (level >= 0 && level < static_cast<int>(metadata_.levelDimensions.size())) {
        return metadata_.levelDimensions[level];
    }
    return cv::Size(0, 0);
}

double OpenCVWSILoader::getLevelDownsample(int level) const {
    if (level >= 0 && level < static_cast<int>(metadata_.levelDownsamples.size())) {
        return metadata_.levelDownsamples[level];
    }
    return 1.0;
}

int OpenCVWSILoader::getBestLevelForDownsample(double downsample) const {
    int bestLevel = 0;
    double minDiff = std::abs(metadata_.levelDownsamples[0] - downsample);
    
    for (int i = 1; i < static_cast<int>(metadata_.levelDownsamples.size()); ++i) {
        double diff = std::abs(metadata_.levelDownsamples[i] - downsample);
        if (diff < minDiff) {
            minDiff = diff;
            bestLevel = i;
        }
    }
    
    return bestLevel;
}

void OpenCVWSILoader::buildPyramid() {
    pyramidLevels_.clear();
    
    cv::Mat currentLevel = fullImage_.clone();
    pyramidLevels_.push_back(currentLevel);
    
    // Build 4 levels of pyramid (downsampling by 2 each time)
    for (int i = 1; i < 4; ++i) {
        cv::Mat nextLevel;
        cv::pyrDown(currentLevel, nextLevel);
        pyramidLevels_.push_back(nextLevel);
        currentLevel = nextLevel;
        
        // Stop if image becomes too small
        if (nextLevel.cols < 64 || nextLevel.rows < 64) {
            break;
        }
    }
}

cv::Mat OpenCVWSILoader::extractRegion(const cv::Mat& source, int x, int y, int width, int height) const {
    // Clamp coordinates to image bounds
    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(source.cols, x + width);
    int y2 = std::min(source.rows, y + height);
    
    if (x1 >= x2 || y1 >= y2) {
        return cv::Mat();
    }
    
    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    return source(roi).clone();
}



} // namespace wsi_io