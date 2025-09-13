#include "wsi/WSILoaderExtensions.h"
#include <iostream>
#include <algorithm>

namespace wsi_io {

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