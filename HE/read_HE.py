#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以馬賽克一塊一塊的方式讀取超大HE CZI，逐塊處理（4000+ tiles），在讀取時直接以0.125倍縮放，
並將H到 HE/picture/H、E到 HE/picture/E（皆為bgr24，3通道uint8）

記憶體安全策略:
- 僅一次讀取一塊瓦片，處理並寫檔後立即釋放
- 使用 CziFile.read_mosaic(..., scale_factor=0.125) 在讀取時就縮小
- 每塊之間明確 del 並 gc.collect()

輸出（每塊）:
- HE/picture/H/HE_20X_ED7_tile{idx}_scale0125_H_bgr.png
- HE/picture/E/HE_20X_ED7_tile{idx}_scale0125_E_bgr.png
"""

import gc
from pathlib import Path
import numpy as np
from aicspylibczi import CziFile
from skimage import color, exposure
import cv2


def create_tissue_mask(rgb_img: np.ndarray, sat_thresh: float = 0.1) -> np.ndarray:
    """
    利用影像的飽和度(Saturation)資訊來生成一個乾淨的組織遮罩，從而徹底清除背景噪聲。
    這個遮罩會明確地告訴我們，影像中的哪些部分是真實的組織，哪些部分是應該被完全忽略的背景。
    """
    if rgb_img is None:
        raise ValueError("rgb_img must not be None")
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image")

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    _, s, _ = cv2.split(hsv)

    # 使用飽和度閾值創建遮罩
    # cv2.threshold returns the threshold value and the thresholded image (mask)
    _, mask = cv2.threshold(s, int(sat_thresh * 255), 255, cv2.THRESH_BINARY)

    return mask.astype(bool)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """將影像轉為uint8以利PNG輸出：
    - 支援 uint8 / uint16 / float 影像。
    - 採用 1-99 百分位做對比拉伸，避免極值影響。
    - 保證不複製過多中間陣列，處理後立刻返回可寫出之資料。
    """
    if img is None:
        raise ValueError("to_uint8: img is None")

    if img.dtype == np.uint8:
        return img

    # 單通道或多通道皆適用：分別對每個通道做拉伸，最後疊回
    if img.ndim == 2:
        low, high = np.percentile(img, (1, 99))
        if high <= low:
            high = img.max()
            low = img.min()
        scaled = exposure.rescale_intensity(img, in_range=(low, high), out_range=(0, 255))
        out = scaled.astype(np.uint8, copy=False)
        del scaled
        return out
    elif img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            ch = img[..., c]
            low, high = np.percentile(ch, (1, 99))
            if high <= low:
                high = ch.max()
                low = ch.min()
            scaled = exposure.rescale_intensity(ch, in_range=(low, high), out_range=(0, 255))
            channels.append(scaled.astype(np.uint8, copy=False))
            del ch, scaled
        out = np.stack(channels, axis=2)
        del channels
        return out
    else:
        raise ValueError(f"to_uint8: unsupported ndim={img.ndim}")


def bgr_to_rgb_if_needed(img: np.ndarray, pixel_type: str) -> np.ndarray:
    """根據 CZI 的 pixel_type 轉換 BGR → RGB（skimage 期望 RGB）。"""
    if img is None:
        raise ValueError("bgr_to_rgb_if_needed: img is None")

    # 僅在最後一維為3（彩色）時考慮轉換
    if img.ndim == 3 and img.shape[-1] == 3:
        if pixel_type and pixel_type.lower().startswith("bgr"):
            # OpenCV 是 BGR；這裡轉為 RGB 供 skimage 使用
            return img[..., ::-1]
    return img


def color_deconvolution_he(rgb: np.ndarray) -> tuple:
    """執行H&E色彩分解。
    回傳 (H_gray, E_gray)，範圍為 uint8 0~255（白=淺，黑=濃）。
    作法：RGB → OD → HED分解 → 將H/E通道轉回強度影像並拉伸到8位。
    """
    if rgb is None:
        raise ValueError("color_deconvolution_he: rgb is None")
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("color_deconvolution_he 需要RGB三通道影像")

    # skimage 的 HED 矩陣
    hed_matrix = color.hed_from_rgb
    # 將 RGB 轉為 HED（OD 空間）
    # separate_stains 會先做對數轉換至 OD 後投影
    hed = color.separate_stains(rgb.astype(np.float32) / 255.0, hed_matrix)

    # 取出 H 與 E 的濃度（OD 越大代表染色越濃）
    H_od = hed[..., 0]
    E_od = hed[..., 1]

    # 將 OD 轉回強度（傳統顯示上染色越濃越暗）: I = exp(-OD)，再轉成 8-bit
    H_int = np.exp(-H_od)
    E_int = np.exp(-E_od)

    # 轉換為8位以利輸出；為避免極值干擾，使用百分位拉伸
    H_u8 = to_uint8(H_int)
    E_u8 = to_uint8(E_int)

    # 清理中間陣列
    del hed, H_od, E_od, H_int, E_int
    gc.collect()

    return H_u8, E_u8


def save_png(path: Path, img: np.ndarray, color_bgr: bool = False):
    """以PNG寫檔；若color_bgr=True，代表輸入為BGR彩色。
    灰階直接以單通道寫出。
    """
    if img is None:
        raise ValueError("save_png: img is None")

    path.parent.mkdir(parents=True, exist_ok=True)

    to_write = img
    if img.ndim == 3 and img.shape[-1] == 3 and color_bgr:
        to_write = img  # 已是BGR
    elif img.ndim == 3 and img.shape[-1] == 3 and not color_bgr:
        # RGB → BGR 以符合 OpenCV 寫檔
        to_write = img[..., ::-1]

    ok = cv2.imwrite(str(path), to_write)
    if not ok:
        raise IOError(f"寫入PNG失敗: {path}")


def format_scale_token(scale: float) -> str:
    """將 0.125 → '0125'，0.25 → '0250' 等方便檔名辨識。
    依照範例使用 scale*1000 四位數補零。
    """
    return f"{int(scale*1000):04d}"


def process_all_tiles(scale: float = 0.125, max_tiles: int | None = None):
    czi_path = Path("E:/Class/tsgh/picture/whole_size/HE_20X_ED7.czi")
    out_root = Path("E:/Class/tsgh/HE/picture/")

    if not czi_path.exists():
        raise FileNotFoundError(f"找不到檔案: {czi_path}")

    czi = None
    try:
        czi = CziFile(str(czi_path))
        if not czi.is_mosaic():
            raise RuntimeError("此CZI不是馬賽克格式，無法以tile方式處理")

        bboxes = czi.get_all_mosaic_tile_bounding_boxes()
        if not bboxes:
            raise RuntimeError("未取得馬賽克瓦片資訊")

        bbox_items = list(bboxes.items())
        total = len(bbox_items)
        if max_tiles is not None:
            total = min(total, max_tiles)
        scale_token = format_scale_token(scale)

        print(f"將以 {scale:.4f}x 縮放處理瓦片，共 {total} 塊，拼接到一張最終畫布。")

        # 計算全域畫布大小
        print("計算全域畫布大小...")
        all_positions = [(bbox.x, bbox.y, bbox.w, bbox.h) for _, bbox in bbox_items[:total]]

        global_min_x = min(pos[0] for pos in all_positions)
        global_min_y = min(pos[1] for pos in all_positions)
        global_max_x = max(pos[0] + pos[2] for pos in all_positions)
        global_max_y = max(pos[1] + pos[3] for pos in all_positions)

        canvas_w = int((global_max_x - global_min_x) * scale)
        canvas_h = int((global_max_y - global_min_y) * scale)

        print(f"最終畫布大小: {canvas_w} x {canvas_h}")
        print(f"估計記憶體需求: {canvas_w * canvas_h * 2 / (1024**3):.2f} GB")

        # 創建最終畫布
        print("創建最終畫布...")
        final_h = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        final_e = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # 逐個處理所有瓦片，直接放到最終畫布上
        for idx in range(total):
            tile_id, bbox = bbox_items[idx]
            tile_img = None
            rgb = None
            rgb_u8 = None
            H_u8 = None
            E_u8 = None

            try:
                region = (bbox.x, bbox.y, bbox.w, bbox.h)

                try:
                    tile_img = czi.read_mosaic(region, scale_factor=scale, C=0)
                except Exception:
                    tile_img = czi.read_mosaic(region, scale_factor=scale)

                if tile_img is None or tile_img.size == 0:
                    print(f"[跳過] 瓦片 {idx}: 讀取為空")
                    continue

                if hasattr(tile_img, 'squeeze'):
                    tile_img = tile_img.squeeze()

                # 轉為RGB供色彩分解
                rgb = bgr_to_rgb_if_needed(tile_img, czi.pixel_type)
                rgb_u8 = to_uint8(rgb)

                # 色彩分解
                H_u8, E_u8 = color_deconvolution_he(rgb_u8)

                # 創建並應用組織遮罩
                tissue_mask = create_tissue_mask(rgb_u8, sat_thresh=0.1)
                H_u8[~tissue_mask] = 255  # 背景設為白色
                E_u8[~tissue_mask] = 255  # 背景設為白色

                # 計算在最終畫布上的位置
                canvas_x = int((bbox.x - global_min_x) * scale)
                canvas_y = int((bbox.y - global_min_y) * scale)

                # 確保不超出邊界
                end_x = min(canvas_x + H_u8.shape[1], canvas_w)
                end_y = min(canvas_y + H_u8.shape[0], canvas_h)
                tile_w = end_x - canvas_x
                tile_h = end_y - canvas_y

                # 直接放到最終畫布上
                if tile_w > 0 and tile_h > 0:
                    final_h[canvas_y:end_y, canvas_x:end_x] = H_u8[:tile_h, :tile_w]
                    final_e[canvas_y:end_y, canvas_x:end_x] = E_u8[:tile_h, :tile_w]

                # 釋放當前瓦片資源
                del tile_img, rgb, rgb_u8, H_u8, E_u8

                # 每100個瓦片報告一次進度
                if (idx + 1) % 100 == 0:
                    print(f"進度: {idx + 1}/{total} 瓦片已處理")

            except Exception as e:
                print(f"[錯誤] 瓦片 {idx} 處理失敗: {e}")
                # 確保清理
                for v in [tile_img, rgb, rgb_u8, H_u8, E_u8]:
                    try:
                        if v is not None:
                            del v
                    except Exception:
                        pass
                continue

        # 輸出最終結果
        print("正在儲存最終畫布...")
        h_path = out_root / f"HE_20X_ED7_final_scale{scale_token}_H.png"
        e_path = out_root / f"HE_20X_ED7_final_scale{scale_token}_E.png"

        save_png(h_path, final_h, color_bgr=False)
        save_png(e_path, final_e, color_bgr=False)

        print(f"✓ 最終畫布已儲存:")
        print(f"  H通道: {h_path}")
        print(f"  E通道: {e_path}")
        print(f"  尺寸: {canvas_w} x {canvas_h}")
        print(f"  處理瓦片: {total}")

        # 清理最終資源
        del final_h, final_e

        print("全部瓦片處理完成。")

    finally:
        if czi is not None:
            del czi
        gc.collect()


if __name__ == "__main__":
    # 將所有瓦片直接拼接到一張最終畫布，每塊縮放 0.125 倍
    process_all_tiles(scale=0.125)
    print("完成：已將所有瓦片拼接成最終H/E灰階PNG。")
