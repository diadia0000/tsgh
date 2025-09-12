#include "core/CellRegistration.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>
using namespace std;
void printUsage(const char* programName) {
    cout << "Cell Image Registration System\n";
    cout << "=============================\n\n";
    cout << "Usage: " << programName << " --input <tiff_dir> --output <output_dir> [options]\n";
    cout << "\nOptions:\n";
    cout << "  --input <directory>         Input directory with TIFF files (required)\n";
    cout << "  --output <directory>        Output directory (required)\n";
    cout << "  --gpu                       Enable GPU acceleration\n";
    cout << "  --help                      Show this help message\n";
    cout << "\nExample:\n";
    cout << "  " << programName << " --input picture/tiff/ --output picture/output/ --gpu\n";
    cout << "\nExpected input files: *HE*.tiff, *Her2*.tiff, *DISH*.tiff\n";
}



int main(int argc, char* argv[]) {
    cout << "Cell Image Registration System\n";
    cout << "=============================\n\n";

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    string inputDir, outputDir;
    bool useGpu = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            inputDir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--gpu") {
            useGpu = true;
        }
    }
    
    if (inputDir.empty() || outputDir.empty()) {
        cerr << "Error: Both --input and --output directories are required\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Ensure directories end with separator
    if (inputDir.back() != '/' && inputDir.back() != '\\') inputDir += "/";
    if (outputDir.back() != '/' && outputDir.back() != '\\') outputDir += "/";
    
    try {
        cell_registration::CellRegistration registrator;
        registrator.setGpuEnabled(useGpu);

        cout << "Configuration:\n";
        cout << "  Input Directory: " << inputDir << "\n";
        cout << "  Output Directory: " << outputDir << "\n";
        cout << "  GPU Acceleration: " << (useGpu ? "Enabled" : "Disabled") << "\n\n";

        auto startTime = chrono::high_resolution_clock::now();
        
        if (!registrator.performRegistration(inputDir, outputDir)) {
            cerr << "Error: Registration failed\n";
            return 1;
        }

        auto endTime = chrono::high_resolution_clock::now();
        auto totalTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);

        cout << "\n[SUCCESS] Cell image registration completed in "
             << totalTime.count() << " ms\n";

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}