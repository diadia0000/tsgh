import cv2
import numpy as np
from aicspylibczi import CziFile
from pathlib import Path
import gc

def read_czi_tile(filepath, tile_index=0):
    """
    讀取CZI馬賽克檔案的第一個圖塊並回傳灰階影像。
    """
    print(f"正在讀取檔案: {filepath.name}")
    czi = None
    try:
        czi = CziFile(filepath)
        if not czi.is_mosaic():
            print(f"警告: {filepath.name} 不是馬賽克檔案。")
            # 讀取第一個場景
            image = czi.read_image(scene=0).squeeze()
        else:
            # 獲取第一個圖塊的邊界框
            tile_bboxes = czi.get_all_mosaic_tile_bounding_boxes()
            if not tile_bboxes:
                raise ValueError("馬賽克影像不包含任何圖塊資訊")
            
            # 確保圖塊索引有效
            if tile_index >= len(tile_bboxes):
                print(f"警告: 圖塊索引 {tile_index} 超出範圍，將使用第一個圖塊。")
                tile_index = 0

            tile_bbox_key = list(tile_bboxes.keys())[tile_index]
            first_tile_bbox = tile_bboxes[tile_bbox_key]
            region_tuple = (first_tile_bbox.x, first_tile_bbox.y, first_tile_bbox.w, first_tile_bbox.h)
            
            print(f"  - 正在讀取圖塊 {tile_index}，邊界框: {region_tuple}")
            # 讀取圖塊
            image = czi.read_mosaic(region_tuple, scale_factor=1.0, C=0).squeeze()

        # 確保影像是 8-bit
        if image.dtype != np.uint8:
            print(f"  - 影像類型為 {image.dtype}，將轉換為 8-bit (0-255)。")
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 轉換為灰階
        if len(image.shape) == 3 and image.shape[2] == 3: # BGR
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image # 假設已經是灰階

        print(f"  - ✓ 讀取與轉換成功，影像形狀: {gray_image.shape}")
        return gray_image

    except Exception as e:
        print(f"  - ✗ 讀取失敗: {e}")
        return None
    finally:
        if czi:
            del czi
            gc.collect()

def find_transformation(image1, image2):
    """
    在兩個影像之間尋找最佳的仿射變換矩陣。
    返回 (變換矩陣, 好的匹配點數量)。
    """
    # 影像增強 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image1 = clahe.apply(image1)
    enhanced_image2 = clahe.apply(image2)

    # 初始化SIFT偵測器
    sift = cv2.SIFT_create(contrastThreshold=0.03)

    # 尋找關鍵點與描述子
    kp1, des1 = sift.detectAndCompute(enhanced_image1, None)
    kp2, des2 = sift.detectAndCompute(enhanced_image2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0

    # FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    all_matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in all_matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 10
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0)

        # 計算內點 (inliers) 數量，作為匹配品質的指標
        inlier_count = np.sum(mask) if mask is not None else 0
        if inlier_count >= MIN_MATCH_COUNT:
            return M, inlier_count

    return None, len(good_matches)


def main():
    """
    主函式：讀取參考影像，並在目標影像的所有圖塊中搜索最佳匹配。
    """
    picture_dir = Path("E:/Class/tsgh/picture")
    
    ref_file = picture_dir / "P2525729F_HE_region.czi"
    target_files = {
        "HER2": picture_dir / "P2525729F_HER2_region.czi",
        "DISH": picture_dir / "P2525729F_DISH_region.czi"
    }

    print(f"讀取參考影像: {ref_file.name}")
    ref_image = read_czi_tile(ref_file, 0)
    if ref_image is None:
        print("無法讀取參考影像，程序中止。")
        return

    for target_name, target_path in target_files.items():
        print(f"\n{'='*60}")
        print(f"正在處理目標檔案: {target_name}")
        if not target_path.exists():
            print(f"找不到檔案: {target_path}")
            continue

        best_match_count = 0
        best_matrix = None
        best_tile_index = -1
        best_tile_image = None

        try:
            target_czi = CziFile(target_path)
            if not target_czi.is_mosaic:
                print(f"{target_name} 不是馬賽克檔案，跳過。")
                continue

            tile_bboxes = target_czi.get_all_mosaic_tile_bounding_boxes()
            num_tiles = len(tile_bboxes)
            print(f"在 {target_name} 中找到 {num_tiles} 個圖塊，開始遍歷搜索...")

            for i, (tile_key, bbox) in enumerate(tile_bboxes.items()):
                print(f"  - 正在比對圖塊 {i+1}/{num_tiles}...")

                # 讀取當前圖塊
                region_tuple = (bbox.x, bbox.y, bbox.w, bbox.h)
                current_tile_img_raw = target_czi.read_mosaic(region_tuple, scale_factor=1.0, C=0).squeeze()

                # 預處理
                if current_tile_img_raw.dtype != np.uint8:
                    current_tile_img = cv2.normalize(current_tile_img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                else:
                    current_tile_img = current_tile_img_raw

                if len(current_tile_img.shape) == 3:
                    current_tile_img = cv2.cvtColor(current_tile_img, cv2.COLOR_BGR2GRAY)

                # 尋找變換
                matrix, match_count = find_transformation(current_tile_img, ref_image)

                if matrix is not None and match_count > best_match_count:
                    best_match_count = match_count
                    best_matrix = matrix
                    best_tile_index = i
                    best_tile_image = current_tile_img_raw # 保存原始影像用於最終變換
                    print(f"    ✓ 找到新的最佳匹配！圖塊索引: {i}, 匹配點數: {match_count}")

        finally:
            if 'target_czi' in locals():
                del target_czi
            gc.collect()

        # 處理最終結果
        print(f"\n對 {target_name} 的搜索完成。")
        if best_matrix is not None:
            print(f"最佳匹配位於圖塊 {best_tile_index}，擁有 {best_match_count} 個可靠匹配點。")

            h, w = ref_image.shape
            aligned_image = cv2.warpAffine(best_tile_image, best_matrix, (w, h))

            output_filename = f"aligned_{target_name}_to_HE_best_match.png"
            cv2.imwrite(output_filename, aligned_image)
            print(f"✓ 最終對齊影像已儲存至: {output_filename}")
        else:
            print(f"✗ 在 {target_name} 的所有圖塊中，均未找到足夠的匹配點來進行可靠的對齊。")


if __name__ == "__main__":
    main()
