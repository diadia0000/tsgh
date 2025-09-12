#include "WSILoader.h"
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

// CZILoader implementation
bool CZILoader::open(const std::string& filepath) {
    filepath_ = filepath;
    
    // For MVP, use OpenCV fallback for CZI files
    // In production, this would use a proper CZI library like libCZI
    fallbackLoader_ = std::make_unique<OpenCVWSILoader>();
    
    if (!fallbackLoader_->open(filepath)) {
        std::cerr << "Warning: CZI format not fully supported, using OpenCV fallback" << std::endl;
        return false;
    }
    
    metadata_ = fallbackLoader_->getMetadata();
    metadata_.format = WSIFormat::CZI;
    metadata_.vendor = "Carl Zeiss";
    
    isOpen_ = true;
    return true;
}

void CZILoader::close() {
    if (fallbackLoader_) {
        fallbackLoader_->close();
        fallbackLoader_.reset();
    }
    isOpen_ = false;
}

cv::Mat CZILoader::readRegion(int x, int y, int width, int height, int level) {
    if (fallbackLoader_) {
        return fallbackLoader_->readRegion(x, y, width, height, level);
    }
    return cv::Mat();
}

cv::Mat CZILoader::readTile(const TileInfo& tile) {
    if (fallbackLoader_) {
        return fallbackLoader_->readTile(tile);
    }
    return cv::Mat();
}

cv::Mat CZILoader::getThumbnail(int maxSize) {
    if (fallbackLoader_) {
        return fallbackLoader_->getThumbnail(maxSize);
    }
    return cv::Mat();
}

cv::Size CZILoader::getLevelDimensions(int level) const {
    if (fallbackLoader_) {
        return fallbackLoader_->getLevelDimensions(level);
    }
    return cv::Size(0, 0);
}

double CZILoader::getLevelDownsample(int level) const {
    if (fallbackLoader_) {
        return fallbackLoader_->getLevelDownsample(level);
    }
    return 1.0;
}

int CZILoader::getBestLevelForDownsample(double downsample) const {
    if (fallbackLoader_) {
        return fallbackLoader_->getBestLevelForDownsample(downsample);
    }
    return 0;
}

// TiledWSIReader implementation
TiledWSIReader::TiledWSIReader(std::unique_ptr<WSILoader> loader) 
    : loader_(std::move(loader)) {
}

std::vector<TileInfo> TiledWSIReader::generateTiles(int level) const {
    std::vector<TileInfo> tiles;
    
    if (!loader_ || !loader_->isOpen()) {
        return tiles;
    }
    
    cv::Size levelSize = loader_->getLevelDimensions(level);
    
    for (int y = 0; y < levelSize.height; y += tileHeight_ - overlap_) {
        for (int x = 0; x < levelSize.width; x += tileWidth_ - overlap_) {
            TileInfo tile;
            tile.x = x;
            tile.y = y;
            tile.width = std::min(tileWidth_, levelSize.width - x);
            tile.height = std::min(tileHeight_, levelSize.height - y);
            tile.level = level;
            
            tiles.push_back(tile);
        }
    }
    
    return tiles;
}

cv::Mat TiledWSIReader::readNextTile() {
    if (!hasMoreTiles()) {
        return cv::Mat();
    }
    
    if (allTiles_.empty()) {
        allTiles_ = generateTiles(0);
    }
    
    TileInfo tile = allTiles_[currentTileIndex_++];
    cv::Mat tileImage = loader_->readTile(tile);
    
    updateMemoryUsage(tileImage, true);
    
    return tileImage;
}

bool TiledWSIReader::hasMoreTiles() const {
    if (allTiles_.empty()) {
        return true; // Haven't generated tiles yet
    }
    return currentTileIndex_ < allTiles_.size();
}

void TiledWSIReader::reset() {
    currentTileIndex_ = 0;
    currentMemoryBytes_ = 0;
}

std::vector<cv::Mat> TiledWSIReader::readTileBatch(const std::vector<TileInfo>& tiles) {
    std::vector<cv::Mat> batch;
    batch.reserve(tiles.size());
    
    for (const auto& tile : tiles) {
        cv::Mat tileImage = loader_->readTile(tile);
        if (!tileImage.empty()) {
            batch.push_back(tileImage);
            updateMemoryUsage(tileImage, true);
            
            // Check memory limit
            if (currentMemoryBytes_ > maxMemoryBytes_) {
                std::cerr << "Warning: Memory limit exceeded, stopping batch read" << std::endl;
                break;
            }
        }
    }
    
    return batch;
}

void TiledWSIReader::updateMemoryUsage(const cv::Mat& image, bool add) {
    size_t imageBytes = image.total() * image.elemSize();
    
    if (add) {
        currentMemoryBytes_ += imageBytes;
    } else {
        currentMemoryBytes_ = (currentMemoryBytes_ > imageBytes) ? 
                             currentMemoryBytes_ - imageBytes : 0;
    }
}

} // namespace wsi_io