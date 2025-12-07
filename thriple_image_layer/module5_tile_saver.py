"""
Module 5: Tile-based Image Warping and Saving
改進版：直接輸出 tiles，避免拼接時的記憶體爆炸

解決 VALIS 的問題：
1. VALIS 使用 tile-based 計算位移場
2. 但最後要拼接成完整圖像（記憶體爆炸！）
3. 這個模組改為：計算一個 tile，輸出一個 tile
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pyvips
from valis import registration, slide_io
import json
import time


class TileSaver:
    """
    逐 tile 變換並直接儲存，避免記憶體爆炸
    """

    def __init__(self, registrar, output_dir, level=0, tile_wh=4096, batch_size=16):
        self.registrar = registrar
        self.output_dir = Path(output_dir)
        self.level = level
        self.tile_wh = tile_wh
        self.batch_size = batch_size

        # 設置 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")

    def process_and_save_tiles(self, slide_name, non_rigid=True):
        """
        處理並儲存單個 slide 的所有 tiles

        優勢：
        1. 不需要拼接 - 直接輸出 tiles
        2. 記憶體使用恆定 - 只需要處理當前 batch
        3. 可以暫停和恢復 - tiles 獨立
        """

        slide_obj = self.registrar.slide_dict[slide_name]
        width, height = slide_obj.slide_dimensions_wh[self.level]

        print(f"\n處理 {slide_name}")
        print(f"Level {self.level} 尺寸: {width} x {height}")

        # 創建輸出目錄
        slide_output_dir = self.output_dir / f"{slide_name}_tiles_lv{self.level}"
        slide_output_dir.mkdir(parents=True, exist_ok=True)

        # 計算 tiles
        tiles = []
        for y in range(0, height, self.tile_wh):
            for x in range(0, width, self.tile_wh):
                w = min(self.tile_wh, width - x)
                h = min(self.tile_wh, height - y)
                tiles.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'row': y // self.tile_wh,
                    'col': x // self.tile_wh
                })

        total_tiles = len(tiles)
        print(f"總共 {total_tiles} 個 tiles")

        # 準備變換矩陣
        M_scaled = self._get_scaled_transform_matrix(slide_obj)
        M_gpu = torch.from_numpy(M_scaled[:2]).unsqueeze(0).float().to(self.device)

        # 準備位移場
        bk_dxdy_gpu = None
        if non_rigid and hasattr(slide_obj, 'bk_dxdy') and slide_obj.bk_dxdy is not None:
            bk_dxdy_np = slide_obj.bk_dxdy
            if isinstance(bk_dxdy_np, (list, tuple)):
                bk_dxdy_np = np.stack(bk_dxdy_np, axis=0)
            elif bk_dxdy_np.ndim == 3 and bk_dxdy_np.shape[2] == 2:
                bk_dxdy_np = bk_dxdy_np.transpose(2, 0, 1)

            if bk_dxdy_np.ndim == 3:
                bk_dxdy_gpu = torch.from_numpy(bk_dxdy_np).unsqueeze(0).float().to(self.device)
            else:
                bk_dxdy_gpu = torch.from_numpy(bk_dxdy_np).float().to(self.device)

        # 計算縮放比例
        level_scale_non_rigid = 1.0
        if bk_dxdy_gpu is not None:
            reg_img_shape_rc = slide_obj.processed_img_shape_rc
            level_scale_non_rigid = reg_img_shape_rc[0] / height

        # 處理 tiles
        start_time = time.time()
        for i in range(0, total_tiles, self.batch_size):
            batch_tiles = tiles[i:i+self.batch_size]

            # 讀取並變換 batch
            warped_tiles = self._process_batch(
                slide_obj, batch_tiles,
                M_gpu, bk_dxdy_gpu, level_scale_non_rigid
            )

            # 直接儲存每個 tile
            for tile_info, warped_tile in zip(batch_tiles, warped_tiles):
                self._save_tile(slide_output_dir, tile_info, warped_tile)

            # 顯示進度
            processed = min(i + self.batch_size, total_tiles)
            elapsed = time.time() - start_time
            speed = processed / elapsed
            eta = (total_tiles - processed) / speed
            print(f"進度: {processed}/{total_tiles} ({processed/total_tiles*100:.1f}%) | "
                  f"速度: {speed:.2f} tiles/s | ETA: {eta:.0f}s", end='\r')

        print()

        # 儲存 metadata
        metadata = {
            'slide_name': slide_name,
            'level': self.level,
            'original_size': [width, height],
            'tile_size': self.tile_wh,
            'total_tiles': total_tiles,
            'grid_size': [
                (height + self.tile_wh - 1) // self.tile_wh,
                (width + self.tile_wh - 1) // self.tile_wh
            ],
            'non_rigid': non_rigid,
            'tiles': tiles
        }

        with open(slide_output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ 完成！Tiles 儲存於: {slide_output_dir}")
        print(f"✓ Metadata 儲存於: {slide_output_dir / 'metadata.json'}")

        return slide_output_dir

    def _get_scaled_transform_matrix(self, slide_obj):
        """獲取縮放後的變換矩陣"""
        M = slide_obj.M.copy()
        level_scale = slide_obj.slide_dimensions_wh[0][0] / slide_obj.slide_dimensions_wh[self.level][0]

        S = np.array([
            [1 / level_scale, 0, 0],
            [0, 1 / level_scale, 0],
            [0, 0, 1]
        ])
        S_inv = np.array([
            [level_scale, 0, 0],
            [0, level_scale, 0],
            [0, 0, 1]
        ])

        return S @ M @ S_inv

    def _process_batch(self, slide_obj, batch_tiles, M_gpu, bk_dxdy_gpu, level_scale):
        """處理一個 batch 的 tiles"""

        with torch.no_grad():
            # 讀取 tiles
            tile_images = []
            for tile_info in batch_tiles:
                tile = slide_obj.reader.slide2image(
                    level=self.level,
                    xywh=(tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h'])
                )
                tile_images.append(tile)

            # 轉換為 tensor
            batch_tensors = []
            for tile in tile_images:
                if tile.ndim == 2:
                    t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
                else:
                    t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
                batch_tensors.append(t)

            tiles_tensor = torch.cat(batch_tensors, dim=0).float().to(self.device)
            B, C, H, W = tiles_tensor.shape

            # 步驟1: 剛性變換
            affine_grid = F.affine_grid(M_gpu[:, :2], size=(B, C, H, W), align_corners=False)
            affine_warped = F.grid_sample(tiles_tensor, affine_grid,
                                         mode='bilinear', padding_mode='zeros', align_corners=False)

            # 步驟2: 非剛性變換
            if bk_dxdy_gpu is not None:
                batch_fields = []
                for tile_info in batch_tiles:
                    x, y, w, h = tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h']

                    field_x = int(x * level_scale)
                    field_y = int(y * level_scale)
                    field_w = int(w * level_scale)
                    field_h = int(h * level_scale)

                    x_start = max(0, min(field_x, bk_dxdy_gpu.shape[3] - 1))
                    y_start = max(0, min(field_y, bk_dxdy_gpu.shape[2] - 1))
                    x_end = min(bk_dxdy_gpu.shape[3], field_x + max(1, field_w))
                    y_end = min(bk_dxdy_gpu.shape[2], field_y + max(1, field_h))

                    local_field = bk_dxdy_gpu[:, :, y_start:y_end, x_start:x_end]
                    local_field_resized = F.interpolate(
                        local_field, size=(H, W), mode='bilinear', align_corners=False
                    )
                    batch_fields.append(local_field_resized)

                fields_tensor = torch.cat(batch_fields, dim=0) * (1.0 / level_scale)

                # 創建變形網格
                y_coords = torch.arange(H, device=self.device).view(H, 1).expand(H, W)
                x_coords = torch.arange(W, device=self.device).view(1, W).expand(H, W)

                displaced_x = x_coords + fields_tensor[:, 0]
                displaced_y = y_coords + fields_tensor[:, 1]

                norm_x = 2.0 * displaced_x / (W - 1) - 1.0
                norm_y = 2.0 * displaced_y / (H - 1) - 1.0
                non_rigid_grid = torch.stack([norm_x, norm_y], dim=-1)

                warped_batch = F.grid_sample(affine_warped, non_rigid_grid,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            else:
                warped_batch = affine_warped

            # 轉回 numpy
            warped_np = warped_batch.cpu().numpy()

            results = []
            for i, tile in enumerate(tile_images):
                if tile.ndim == 2:
                    result = warped_np[i, 0]
                else:
                    result = warped_np[i].transpose(1, 2, 0)
                results.append(result)

            return results

    def _save_tile(self, output_dir, tile_info, tile_data):
        """儲存單個 tile"""
        filename = f"tile_r{tile_info['row']:04d}_c{tile_info['col']:04d}.tif"
        filepath = output_dir / filename

        # 轉換為 uint8
        if tile_data.dtype != np.uint8:
            tile_data = np.clip(tile_data, 0, 255).astype(np.uint8)

        # 使用 pyvips 儲存
        vips_img = pyvips.Image.new_from_array(tile_data)
        vips_img.write_to_file(str(filepath), compression='lzw')


def reassemble_tiles(tile_dir, output_file=None):
    """
    （可選）從 tiles 重新組合成完整圖像

    注意：這個函數仍然需要大量記憶體
    但至少你可以選擇：
    1. 只使用 tiles（例如給其他軟體）
    2. 後續再組合（當有更多記憶體時）
    """
    tile_dir = Path(tile_dir)

    # 讀取 metadata
    with open(tile_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"重新組合 {metadata['slide_name']}")
    print(f"Grid: {metadata['grid_size'][0]} x {metadata['grid_size'][1]}")

    if output_file is None:
        output_file = tile_dir.parent / f"{metadata['slide_name']}_reassembled_lv{metadata['level']}.tif"

    # 逐行組合
    rows = []
    for row_idx in range(metadata['grid_size'][0]):
        row_tiles = []
        for col_idx in range(metadata['grid_size'][1]):
            filename = f"tile_r{row_idx:04d}_c{col_idx:04d}.tif"
            filepath = tile_dir / filename

            if filepath.exists():
                tile = pyvips.Image.new_from_file(str(filepath))
                row_tiles.append(tile)

        if row_tiles:
            row_img = row_tiles[0]
            for tile in row_tiles[1:]:
                row_img = row_img.join(tile, 'horizontal')
            rows.append(row_img)

    # 組合所有行
    if rows:
        final_img = rows[0]
        for row in rows[1:]:
            final_img = final_img.join(row, 'vertical')

        # 儲存
        final_img.write_to_file(
            str(output_file),
            tile=True,
            pyramid=True,
            compression='jpeg',
            Q=90,
            bigtiff=True
        )

        print(f"✓ 重新組合完成: {output_file}")
        return output_file

    return None


if __name__ == "__main__":
    # 初始化 JVM
    try:
        slide_io.init_jvm()
    except:
        pass

    try:
        # 載入 registrar
        output_dir = Path(r"H:\tsgh\thriple_image_layer\output")
        pickle_path = output_dir / "Transform_Params" / "data" / "Transform_Params_registrar.pickle"
        registrar = registration.load_registrar(str(pickle_path))

        # 創建 TileSaver
        tile_saver = TileSaver(
            registrar=registrar,
            output_dir=output_dir / "tiles",
            level=1,  # 使用 level 1 避免記憶體問題
            tile_wh=4096,
            batch_size=16
        )

        # 處理 DISH
        print("=" * 80)
        print("處理 DISH 影像（tile-based 輸出）")
        print("=" * 80)
        dish_tiles_dir = tile_saver.process_and_save_tiles('DISH_40X_2', non_rigid=True)

        # 處理 HER2
        print("\n" + "=" * 80)
        print("處理 HER2 影像（tile-based 輸出）")
        print("=" * 80)
        her2_tiles_dir = tile_saver.process_and_save_tiles('HER2_40X', non_rigid=True)

        print("\n" + "=" * 80)
        print("完成！")
        print("=" * 80)
        print(f"\nDISH tiles: {dish_tiles_dir}")
        print(f"HER2 tiles: {her2_tiles_dir}")
        print("\n你現在可以：")
        print("1. 直接使用這些 tiles（例如給 QuPath, ImageJ 等）")
        print("2. 需要時再用 reassemble_tiles() 組合成完整圖像")
        print("3. 用其他軟體逐 tile 處理（例如深度學習推論）")

    finally:
        try:
            slide_io.kill_jvm()
        except:
            pass

