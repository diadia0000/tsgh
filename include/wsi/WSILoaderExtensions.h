#pragma once

#include "WSILoader.h"
#include <memory>

namespace wsi_io {

// CZI format loader (for your specific .czi files)
class CZILoader : public WSILoader {
public:
    CZILoader() = default;
    ~CZILoader() override = default;

    bool open(const std::string& filepath) override;
    void close() override;
    
    WSIMetadata getMetadata() const override { return metadata_; }
    bool isOpen() const override { return isOpen_; }
    
    cv::Mat readRegion(int x, int y, int width, int height, int level = 0) override;
    cv::Mat readTile(const TileInfo& tile) override;
    cv::Mat getThumbnail(int maxSize = 1024) override;
    
    cv::Size getLevelDimensions(int level) const override;
    double getLevelDownsample(int level) const override;
    int getBestLevelForDownsample(double downsample) const override;

private:
    // CZI-specific implementation would go here
    // For MVP, we'll use OpenCV as fallback
    std::unique_ptr<OpenCVWSILoader> fallbackLoader_;
};

// Memory-efficient tile-based reader
class TiledWSIReader {
public:
    explicit TiledWSIReader(std::unique_ptr<WSILoader> loader);
    ~TiledWSIReader() = default;

    // Configure tiling
    void setTileSize(int width, int height) { tileWidth_ = width; tileHeight_ = height; }
    void setOverlap(int overlap) { overlap_ = overlap; }
    
    // Tile iteration
    std::vector<TileInfo> generateTiles(int level = 0) const;
    cv::Mat readNextTile();
    bool hasMoreTiles() const;
    void reset();
    
    // Batch processing
    std::vector<cv::Mat> readTileBatch(const std::vector<TileInfo>& tiles);
    
    // Memory management
    void setMaxMemoryUsage(size_t maxBytes) { maxMemoryBytes_ = maxBytes; }
    size_t getCurrentMemoryUsage() const { return currentMemoryBytes_; }

private:
    std::unique_ptr<WSILoader> loader_;
    int tileWidth_ = 512;
    int tileHeight_ = 512;
    int overlap_ = 64;
    
    std::vector<TileInfo> allTiles_;
    size_t currentTileIndex_ = 0;
    
    size_t maxMemoryBytes_ = 1024 * 1024 * 1024; // 1GB default
    size_t currentMemoryBytes_ = 0;
    
    void updateMemoryUsage(const cv::Mat& image, bool add = true);
};

} // namespace wsi_io