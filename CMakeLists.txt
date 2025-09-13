cmake_minimum_required(VERSION 3.16)

# 設置 vcpkg 工具鏈
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

project(WSIRegistrationMVP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 設定輸出目錄
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist)

# 針對不同配置設定輸出目錄
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/dist/${OUTPUTCONFIG})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/dist/${OUTPUTCONFIG})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/dist/${OUTPUTCONFIG})
endforeach()

# 尋找OpenCV (包含更多組件以支援四階段配準)
find_package(OpenCV REQUIRED COMPONENTS 
    core imgproc imgcodecs highgui features2d calib3d photo
    xfeatures2d ximgproc objdetect ml flann)

# 包含目錄
include_directories(${OpenCV_INCLUDE_DIRS})

# 收集所有源文件
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE IO_SOURCES "src/io/*.cpp")
file(GLOB_RECURSE GPU_SOURCES "src/gpu/*.cpp")

# 添加WSI配準系統執行檔
add_executable(wsi_registration
    src/main.cpp
    ${CORE_SOURCES}
    ${IO_SOURCES}
    ${GPU_SOURCES}
)

# 尋找OpenMP
find_package(OpenMP REQUIRED)

# 設置包含目錄
target_include_directories(wsi_registration PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 連結OpenCV庫
target_link_libraries(wsi_registration ${OpenCV_LIBRARIES})

# 連結OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(wsi_registration OpenMP::OpenMP_CXX)
endif()

# 尋找CUDA（如果可用）
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    # 檢查是否可以啟用 CUDA 語言
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_compile_definitions(wsi_registration PRIVATE CUDA_AVAILABLE)
        target_link_libraries(wsi_registration CUDA::cudart CUDA::cuda_driver)
        message(STATUS "CUDA found and enabled: ${CUDAToolkit_VERSION}")
        
        # 設定 CUDA 架構
        set_property(TARGET wsi_registration PROPERTY CUDA_ARCHITECTURES "75;80;86;89")
    else()
        message(STATUS "CUDA Toolkit found but CUDA compiler not available")
        message(STATUS "Using CPU-only implementation")
    endif()
else()
    message(STATUS "CUDA Toolkit not found, using CPU-only implementation")
endif()

# 編譯選項
target_compile_options(wsi_registration PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /wd4996>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

# 設定預處理器定義
target_compile_definitions(wsi_registration PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    OPENCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR}
    OPENCV_VERSION_MINOR=${OpenCV_VERSION_MINOR}
)

# 顯示配置資訊
message(STATUS "=== WSI Registration System Configuration ===")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBRARIES}")
message(STATUS "CUDA Available: ${CUDAToolkit_FOUND}")
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
endif()
message(STATUS "OpenMP Available: ${OpenMP_CXX_FOUND}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==============================================")