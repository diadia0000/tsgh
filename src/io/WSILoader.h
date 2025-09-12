#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace wsi_io {

enum class WSIFormat {
    SVS,        // Aperio SVS format
    TIFF,       // Tiled TIFF
    CZI,        // Carl Zeiss CZI format
    NDPI,       // Hamamatsu NDPI
    UNKNOWN
};

struct WSIMetadata {
    int width = 0;
    int height = 0;
    int levels = 0;
    double mppX = 0.0;  // Microns per pixel X
    double mppY = 0.0;  // Microns per pixel Y
    WSIFormat format = WSIFormat::UNKNOWN;
    std::string vendor;
    std::string scanDate;
    std::vector<cv::Size> levelDimensions;
    std::vector<double> levelDownsamples;
};

struct TileInfo {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int level = 0;
};

class WSILoader {
public:
    WSILoader() = default;
    virtual ~WSILoader() = default;

    // Main loading functions
    virtual bool open(const std::string& filepath) = 0;
    virtual void close() = 0;
    
    // Metadata access
    virtual WSIMetadata getMetadata() const = 0;
    virtual bool isOpen() const = 0;
    
    // Image reading
    virtual cv::Mat readRegion(int x, int y, int width, int height, int level = 0) = 0;
    virtual cv::Mat readTile(const TileInfo& tile) = 0;
    virtual cv::Mat getThumbnail(int maxSize = 1024) = 0;
    
    // Level operations
    virtual cv::Size getLevelDimensions(int level) const = 0;
    virtual double getLevelDownsample(int level) const = 0;
    virtual int getBestLevelForDownsample(double downsample) const = 0;
    
    // Utility functions
    static WSIFormat detectFormat(const std::string& filepath);
    static std::unique_ptr<WSILoader> createLoader(const std::string& filepath);

protected:
    std::string filepath_;
    WSIMetadata metadata_;
    bool isOpen_ = false;
};

// OpenCV-based loader for standard formats (TIFF, JPEG, etc.)
class OpenCVWSILoader : public WSILoader {
public:
    OpenCVWSILoader() = default;
    ~OpenCVWSILoader() override = default;

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
    cv::Mat fullImage_;
    std::vector<cv::Mat> pyramidLevels_;
    
    void buildPyramid();
    cv::Mat extractRegion(const cv::Mat& source, int x, int y, int width, int height) const;
};

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