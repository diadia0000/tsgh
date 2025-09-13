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



} // namespace wsi_io