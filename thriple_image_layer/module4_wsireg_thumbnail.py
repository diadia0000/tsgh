"""
Module 4: 縮圖生成模組

從 WSIReg 配準結果生成全局疊合縮圖，
支援多種融合模式。
"""
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import tifffile
import pyvips


class BlendMode(Enum):
    """影像融合模式"""
    
    AVERAGE = "average"
    LAPLACIAN = "laplacian"


@dataclass
class ThumbnailConfig:
    """縮圖生成配置"""
    
    # 金字塔層級 (0=最高解析度)
    level: int = 4
    
    # 融合模式
    blend_mode: BlendMode = BlendMode.LAPLACIAN
    
    # 拉普拉斯金字塔層數
    laplacian_levels: int = 5
    
    # 輸出檔名格式
    output_pattern: str = "Merged_Aligned_lv{level}.tiff"


class ThumbnailGenerator:
    """縮圖生成器
    
    從 WSIReg 輸出的 OME-TIFF 生成融合縮圖。
    
    Attributes:
        output_dir: WSIReg 輸出目錄
        config: 縮圖配置
    """
    
    def __init__(
        self,
        output_dir: Path,
        config: Optional[ThumbnailConfig] = None
    ):
        """初始化生成器
        
        Args:
            output_dir: 配準輸出目錄
            config: 縮圖配置
        """
        self.output_dir = Path(output_dir)
        self.config = config or ThumbnailConfig()
    
    def _find_registered_images(self) -> dict[str, Path]:
        """尋找配準後的 OME-TIFF 檔案"""
        image_paths = {}
        
        project_dir = self.output_dir / "thriple_registration"
        if not project_dir.exists():
            project_dir = self.output_dir
        
        patterns = {
            "DISH": "*DISH*.ome.tiff",
            "HER2": "*HER2*.ome.tiff",
        }
        
        for name, pattern in patterns.items():
            matches = list(project_dir.glob(pattern))
            if matches:
                image_paths[name] = matches[0]
        
        return image_paths
    
    def _load_at_level(self, path: Path) -> pyvips.Image:
        """載入指定金字塔層級的影像
        
        Args:
            path: 影像路徑
        
        Returns:
            pyvips 影像物件
        """
        # 嘗試讀取指定層級
        try:
            img = pyvips.Image.new_from_file(
                str(path),
                page=self.config.level,
                access="sequential"
            )
        except pyvips.error.Error:
            # 如果指定層級不存在，讀取完整影像後縮放
            img = pyvips.Image.new_from_file(str(path), access="sequential")
            scale = 1.0 / (2 ** self.config.level)
            img = img.resize(scale)
        
        return img
    
    def _average_blend(
        self,
        img1: pyvips.Image,
        img2: pyvips.Image
    ) -> pyvips.Image:
        """平均融合
        
        Args:
            img1: 第一張影像
            img2: 第二張影像
        
        Returns:
            融合後影像
        """
        return (img1 * 0.5 + img2 * 0.5).cast("uchar")
    
    def _laplacian_blend(
        self,
        img1: pyvips.Image,
        img2: pyvips.Image
    ) -> pyvips.Image:
        """拉普拉斯金字塔融合
        
        保留兩張影像的細節特徵。
        
        Args:
            img1: 第一張影像
            img2: 第二張影像
        
        Returns:
            融合後影像
        """
        levels = self.config.laplacian_levels
        
        # 建立高斯金字塔
        gauss1, gauss2 = [img1], [img2]
        for _ in range(levels):
            gauss1.append(gauss1[-1].shrink(2, 2))
            gauss2.append(gauss2[-1].shrink(2, 2))
        
        # 從最粗層級開始融合
        result = gauss1[-1] * 0.5 + gauss2[-1] * 0.5
        
        # 逐層重建並融合拉普拉斯細節
        for i in range(levels - 1, -1, -1):
            result = result.resize(2, kernel="cubic")
            
            # 計算拉普拉斯層（細節）
            lap1 = gauss1[i] - gauss1[i].shrink(2, 2).resize(2, kernel="cubic")
            lap2 = gauss2[i] - gauss2[i].shrink(2, 2).resize(2, kernel="cubic")
            
            # 融合細節並加回
            result = result + (lap1 * 0.5 + lap2 * 0.5)
        
        return result.cast("uchar")
    
    def _blend_images(
        self,
        img1: pyvips.Image,
        img2: pyvips.Image
    ) -> pyvips.Image:
        """根據配置選擇融合方法
        
        Args:
            img1: 第一張影像
            img2: 第二張影像
        
        Returns:
            融合後影像
        """
        if self.config.blend_mode == BlendMode.AVERAGE:
            return self._average_blend(img1, img2)
        else:
            return self._laplacian_blend(img1, img2)
    
    def _ensure_same_size(
        self,
        img1: pyvips.Image,
        img2: pyvips.Image
    ) -> tuple[pyvips.Image, pyvips.Image]:
        """確保兩張影像尺寸相同
        
        Args:
            img1: 第一張影像
            img2: 第二張影像
        
        Returns:
            調整後的兩張影像
        """
        # 取較小的尺寸
        target_w = min(img1.width, img2.width)
        target_h = min(img1.height, img2.height)
        
        if img1.width != target_w or img1.height != target_h:
            img1 = img1.crop(0, 0, target_w, target_h)
        
        if img2.width != target_w or img2.height != target_h:
            img2 = img2.crop(0, 0, target_w, target_h)
        
        return img1, img2
    
    def run(self) -> Path:
        """執行縮圖生成流程
        
        Returns:
            輸出檔案路徑
        """
        print("=" * 60)
        print("縮圖生成")
        print("=" * 60)
        
        # Step 1: 尋找配準後影像
        print("\n[1/4] 尋找配準後影像...")
        image_paths = self._find_registered_images()
        
        required = {"DISH", "HER2"}
        if not required.issubset(image_paths.keys()):
            raise FileNotFoundError(
                f"找不到必要的配準影像。需要: {required}, 找到: {set(image_paths.keys())}"
            )
        
        for name, path in image_paths.items():
            print(f"  ✓ {name}: {path.name}")
        
        # Step 2: 載入影像
        print(f"\n[2/4] 載入影像 (level={self.config.level})...")
        dish_img = self._load_at_level(image_paths["DISH"])
        her2_img = self._load_at_level(image_paths["HER2"])
        
        print(f"  DISH: {dish_img.width} x {dish_img.height}")
        print(f"  HER2: {her2_img.width} x {her2_img.height}")
        
        # Step 3: 確保尺寸一致並融合
        print(f"\n[3/4] 融合影像 (模式: {self.config.blend_mode.value})...")
        dish_img, her2_img = self._ensure_same_size(dish_img, her2_img)
        merged = self._blend_images(dish_img, her2_img)
        
        print(f"  融合後: {merged.width} x {merged.height}")
        
        # Step 4: 保存輸出
        print("\n[4/4] 保存縮圖...")
        output_name = self.config.output_pattern.format(level=self.config.level)
        output_path = self.output_dir / output_name
        
        merged.write_to_file(
            str(output_path),
            pyramid=True,
            bigtiff=True,
            compression="lzw"
        )
        
        print(f"  ✓ 已保存: {output_path}")
        
        print("\n" + "=" * 60)
        print("縮圖生成完成")
        print("=" * 60)
        
        return output_path


def generate_thumbnail(
    output_dir: Path,
    level: int = 4,
    blend_mode: str = "laplacian"
) -> Path:
    """生成縮圖的便捷函數
    
    提供向後兼容的 API。
    
    Args:
        output_dir: 配準輸出目錄
        level: 金字塔層級
        blend_mode: 融合模式 ("average" 或 "laplacian")
    
    Returns:
        輸出檔案路徑
    """
    config = ThumbnailConfig(
        level=level,
        blend_mode=BlendMode(blend_mode)
    )
    
    generator = ThumbnailGenerator(output_dir, config)
    return generator.run()


if __name__ == "__main__":
    pyvips.cache_set_max(0)
    
    output_dir = Path("/home/sec312/tsgh/thriple_image_layer/output")
    generate_thumbnail(output_dir, level=1)
