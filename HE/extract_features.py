#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
讀取HE_20X_ED7_final_scale0125_E.png，轉換為numpy array並提取特徵值
將特徵值保存為CSV檔案供後續分析使用

特徵包括:
- 基本統計特徵 (mean, std, min, max, percentiles)
- 紋理特徵 (GLCM相關特徵)
- 形態學特徵
- 直方圖特徵
- 空間特徵
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from skimage import feature, measure, morphology
try:
    from skimage.feature.texture import greycomatrix, greycoprops
except ImportError:
    # 如果無法導入GLCM，使用替代方法
    greycomatrix = None
    greycoprops = None
from scipy import stats, ndimage
import gc
from tqdm import tqdm


def load_image_as_array(image_path: Path) -> np.ndarray:
    """載入圖像並轉換為numpy array"""
    print(f"正在載入圖像: {image_path}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"無法載入圖像: {image_path}")

    print(f"圖像尺寸: {img.shape}")
    print(f"數據類型: {img.dtype}")
    print(f"像素值範圍: {img.min()} - {img.max()}")
    return img


def extract_basic_statistics(img: np.ndarray) -> dict:
    """提取基本統計特徵"""
    print("提取基本統計特徵...")
    features = {}

    # 基本統計量
    features['mean'] = float(np.mean(img))
    features['std'] = float(np.std(img))
    features['var'] = float(np.var(img))
    features['min'] = int(np.min(img))
    features['max'] = int(np.max(img))
    features['median'] = float(np.median(img))
    features['range'] = features['max'] - features['min']

    # 百分位數
    percentiles = [5, 10, 25, 75, 90, 95, 99]
    for p in percentiles:
        features[f'percentile_{p}'] = float(np.percentile(img, p))

    # 偏度和峰度
    features['skewness'] = float(stats.skew(img.flatten()))
    features['kurtosis'] = float(stats.kurtosis(img.flatten()))

    # 能量和熵
    hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
    hist = hist + 1e-10  # 避免log(0)
    features['energy'] = float(np.sum(hist ** 2))
    features['entropy'] = float(-np.sum(hist * np.log2(hist)))

    return features


def extract_texture_features_simple(img: np.ndarray) -> dict:
    """提取簡化的紋理特徵"""
    print("提取紋理特徵...")
    features = {}

    # 為了加速計算，對圖像進行下採樣
    sample_ratio = 0.05  # 使用5%的像素進行紋理分析
    h, w = img.shape
    sample_h, sample_w = int(h * sample_ratio), int(w * sample_ratio)

    if sample_h > 100 and sample_w > 100:
        img_sample = cv2.resize(img, (sample_w, sample_h))
        print(f"紋理分析使用採樣尺寸: {img_sample.shape}")

        # 計算方向梯度
        grad_x = cv2.Sobel(img_sample, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_sample, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features['texture_gradient_mean'] = float(np.mean(magnitude))
        features['texture_gradient_std'] = float(np.std(magnitude))

        # 局部標準差 (作為紋理粗糙度的度量)
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(img_sample.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((img_sample.astype(np.float32) - local_mean)**2, -1, kernel)
        local_std = np.sqrt(local_var)

        features['texture_local_std_mean'] = float(np.mean(local_std))
        features['texture_local_std_std'] = float(np.std(local_std))

        # 拉普拉斯算子 (邊緣密度)
        laplacian = cv2.Laplacian(img_sample, cv2.CV_64F)
        features['texture_laplacian_var'] = float(np.var(laplacian))

    else:
        # 如果採樣太小，設置默認值
        for key in ['texture_gradient_mean', 'texture_gradient_std',
                   'texture_local_std_mean', 'texture_local_std_std', 'texture_laplacian_var']:
            features[key] = 0.0

    return features


def extract_morphological_features(img: np.ndarray, threshold_method='otsu') -> dict:
    """提取形態學特徵"""
    print("提取形態學特徵...")
    features = {}

    # 二值化
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        features['otsu_threshold'] = float(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    else:
        thresh = np.mean(img)
        binary = (img > thresh).astype(np.uint8) * 255
        features['otsu_threshold'] = float(thresh)

    # 基本形態學特徵
    total_pixels = binary.size
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    features['binary_white_ratio'] = white_pixels / total_pixels
    features['binary_black_ratio'] = black_pixels / total_pixels

    # 連通組件分析 (使用較小的樣本)
    sample_binary = cv2.resize(binary, (binary.shape[1]//10, binary.shape[0]//10))
    num_labels, labels = cv2.connectedComponents(sample_binary)

    features['connected_components'] = num_labels - 1  # 扣除背景

    # 邊緣檢測
    edges = cv2.Canny(img, 50, 150)
    edge_pixels = np.sum(edges > 0)
    features['edge_density'] = edge_pixels / total_pixels

    return features


def extract_histogram_features(img: np.ndarray) -> dict:
    """提取直方圖特徵"""
    print("提取直方圖特徵...")
    features = {}

    # 計算直方圖
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist_norm = hist / np.sum(hist)  # 歸一化

    # 直方圖統計
    features['hist_peak'] = int(np.argmax(hist))  # 峰值位置
    features['hist_peak_value'] = float(np.max(hist_norm))  # 峰值高度

    # 直方圖的統計矩
    bins = np.arange(256)
    features['hist_mean'] = float(np.sum(bins * hist_norm))
    features['hist_variance'] = float(np.sum(((bins - features['hist_mean'])**2) * hist_norm))
    features['hist_skewness'] = float(np.sum(((bins - features['hist_mean'])**3) * hist_norm) / (features['hist_variance']**1.5) if features['hist_variance'] > 0 else 0)
    features['hist_kurtosis'] = float(np.sum(((bins - features['hist_mean'])**4) * hist_norm) / (features['hist_variance']**2) if features['hist_variance'] > 0 else 0)

    # 直方圖寬度 (有效像素範圍)
    non_zero_bins = np.where(hist > 0)[0]
    if len(non_zero_bins) > 0:
        features['hist_width'] = int(non_zero_bins[-1] - non_zero_bins[0])
        features['hist_left_edge'] = int(non_zero_bins[0])
        features['hist_right_edge'] = int(non_zero_bins[-1])
    else:
        features['hist_width'] = 0
        features['hist_left_edge'] = 0
        features['hist_right_edge'] = 0

    return features


def extract_spatial_features(img: np.ndarray) -> dict:
    """提取空間特徵"""
    print("提取空間特徵...")
    features = {}

    # 使用較小採樣進行空間分析
    sample_img = cv2.resize(img, (500, 500)) if min(img.shape) > 500 else img

    # 計算重心
    moments = cv2.moments(sample_img)
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        features['centroid_x'] = float(cx)
        features['centroid_y'] = float(cy)
    else:
        features['centroid_x'] = float(sample_img.shape[1] / 2)
        features['centroid_y'] = float(sample_img.shape[0] / 2)

    # 方向性分析（使用結構張量）
    Ix = cv2.Sobel(sample_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(sample_img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 結構張量的特徵值
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy

    features['structure_trace_mean'] = float(np.mean(trace))
    features['structure_det_mean'] = float(np.mean(det))

    return features


def extract_all_features(image_path: Path) -> dict:
    """提取所有特徵"""
    print("="*60)
    print("開始特徵提取...")

    # 載入圖像
    img = load_image_as_array(image_path)

    # 初始化特徵字典
    all_features = {}

    # 添加基本資訊
    all_features['image_name'] = image_path.stem
    all_features['image_height'] = img.shape[0]
    all_features['image_width'] = img.shape[1]
    all_features['total_pixels'] = img.size

    try:
        # 基本統計特徵
        basic_features = extract_basic_statistics(img)
        all_features.update(basic_features)

        # 紋理特徵
        texture_features = extract_texture_features_simple(img)
        all_features.update(texture_features)

        # 形態學特徵
        morph_features = extract_morphological_features(img)
        all_features.update(morph_features)

        # 直方圖特徵
        hist_features = extract_histogram_features(img)
        all_features.update(hist_features)

        # 空間特徵
        spatial_features = extract_spatial_features(img)
        all_features.update(spatial_features)

    except Exception as e:
        print(f"特徵提取過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理記憶體
        del img
        gc.collect()

    print(f"特徵提取完成，共提取 {len(all_features)} 個特徵")
    return all_features


def save_features_to_csv(features: dict, output_path: Path):
    """將特徵保存到CSV檔案"""
    print(f"正在保存特徵到: {output_path}")

    # 轉換為DataFrame
    df = pd.DataFrame([features])

    # 保存為CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✓ 特徵已保存到 {output_path}")
    print(f"  - 特徵數量: {len(features)}")
    print(f"  - 檔案大小: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    """主程序"""
    # 設定路徑
    base_path = Path("E:/Class/tsgh/HE/picture")
    image_path = base_path / "HE_20X_ED7_final_scale0125_E.png"
    output_path = base_path / "HE_20X_ED7_final_scale0125_E_features.csv"

    # 檢查輸入檔案
    if not image_path.exists():
        print(f"錯誤: 找不到輸入圖像 {image_path}")
        return

    try:
        # 提取特徵
        features = extract_all_features(image_path)

        # 保存特徵
        save_features_to_csv(features, output_path)

        # 顯示部分特徵預覽
        print("\n特徵預覽 (前10個):")
        print("-" * 40)
        for i, (key, value) in enumerate(list(features.items())[:10]):
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

        print(f"\n總共 {len(features)} 個特徵已成功提取並保存。")

    except Exception as e:
        print(f"程序執行錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
