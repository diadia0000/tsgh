"""
UNet++ 模型可解釋性視覺化工具

使用 Grad-CAM 和 Feature Map 視覺化模型學到的特徵

Author: TSGH AI Team
Date: 2026-02-05
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """載入配置檔案"""
    try:
        from config import config
        return config
    except ImportError:
        raise ImportError("找不到 config.py！")


class GradCAM:
    """
    Grad-CAM 視覺化器
    
    計算模型對特定類別的梯度注意力圖
    
    Attributes:
        model: UNet++ 模型
        target_layer: 目標層 (用於提取特徵和梯度)
        device: 計算設備
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ) -> None:
        """
        初始化 Grad-CAM
        
        Args:
            model: UNet++ 模型
            target_layer: 目標卷積層
            device: 計算設備
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # 註冊 hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """註冊前向和反向 hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        生成 Grad-CAM 熱力圖
        
        Args:
            input_tensor: 輸入影像 Tensor (1, C, H, W)
            target_class: 目標類別 (1=膜)
            
        Returns:
            cam: 熱力圖 (H, W)，值域 [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # 前向傳播
        output = self.model(input_tensor)
        
        # 計算目標類別的分數
        target_score = output[:, target_class, :, :].sum()
        
        # 反向傳播
        self.model.zero_grad()
        target_score.backward()
        
        # 計算 Grad-CAM
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # 全局平均池化梯度
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # 加權特徵圖
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU 並正規化
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # 正規化到 [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


class FeatureMapVisualizer:
    """
    Feature Map 視覺化器
    
    提取並視覺化模型各層的特徵圖
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        """
        初始化視覺化器
        
        Args:
            model: UNet++ 模型
            device: 計算設備
        """
        self.model = model
        self.device = device
        self.feature_maps: Dict[str, torch.Tensor] = {}
        
        # 註冊 hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """註冊前向 hooks 到 encoder 各層"""
        def get_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # 針對 EfficientNet encoder 的各個 block
        for name, module in self.model.encoder.named_children():
            module.register_forward_hook(get_hook(f"encoder_{name}"))
    
    def extract(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        提取各層特徵圖
        
        Args:
            input_tensor: 輸入影像 Tensor (1, C, H, W)
            
        Returns:
            feature_maps: 各層特徵圖字典
        """
        self.model.eval()
        self.feature_maps.clear()
        
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        # 轉換為 numpy
        result = {}
        for name, fmap in self.feature_maps.items():
            # 取第一個樣本，計算通道平均
            fmap_mean = fmap[0].mean(dim=0).cpu().numpy()
            result[name] = fmap_mean
        
        return result


class UNetWatcher:
    """
    UNet++ 模型觀察器
    
    整合 Grad-CAM 和 Feature Map 視覺化
    """
    
    def __init__(
        self,
        model_path: Path,
        encoder_name: str = "efficientnet-b0",
        num_classes: int = 2,
        image_size: Tuple[int, int] = (1024, 1024),
        device: Optional[torch.device] = None,
    ) -> None:
        """
        初始化觀察器
        
        Args:
            model_path: 模型權重路徑
            encoder_name: 編碼器名稱
            num_classes: 類別數量
            image_size: 影像尺寸
            device: 計算設備
        """
        self.image_size = image_size
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        # 建立模型
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
        
        # 載入權重
        self._load_weights(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 建立視覺化器
        # 使用 decoder 的最後一層作為 Grad-CAM 目標
        decoder_blocks = list(self.model.decoder.blocks.children())
        target_layer = decoder_blocks[-1] if decoder_blocks else self.model.segmentation_head[0]
        self.grad_cam = GradCAM(self.model, target_layer, self.device)
        self.feature_viz = FeatureMapVisualizer(self.model, self.device)
        
        # 前處理
        self.transform = self._build_transform()
        
        logger.info(f"觀察器初始化完成，設備: {self.device}")
    
    def _load_weights(self, model_path: Path) -> None:
        """載入模型權重"""
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
    
    def _build_transform(self) -> A.Compose:
        """建立前處理轉換"""
        return A.Compose([
            A.PadIfNeeded(
                min_height=self.image_size[0],
                min_width=self.image_size[1],
                border_mode=0,
                value=255,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """前處理影像"""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        transformed = self.transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def analyze(
        self,
        image_path: Path,
        output_dir: Path,
        show_feature_maps: bool = True,
        n_feature_maps: int = 4,
    ) -> None:
        """
        分析單張影像
        
        Args:
            image_path: 影像路徑
            output_dir: 輸出目錄
            show_feature_maps: 是否顯示特徵圖
            n_feature_maps: 顯示的 encoder 層數
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 讀取影像
        image = io.imread(str(image_path))
        original_size = image.shape[:2]
        
        if image.ndim == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image
        
        # 前處理
        input_tensor = self.preprocess(image_rgb)
        
        # === Grad-CAM ===
        logger.info("生成 Grad-CAM...")
        cam = self.grad_cam.generate(input_tensor, target_class=1)
        
        # 調整 CAM 尺寸以匹配原圖
        cam_resized = self._resize_cam(cam, original_size)
        
        # 生成熱力圖疊加圖
        overlay = self._create_heatmap_overlay(image_rgb, cam_resized)
        
        # 儲存 Grad-CAM
        cam_path = output_dir / f"{image_path.stem}_gradcam.png"
        io.imsave(str(cam_path), overlay)
        logger.info(f"Grad-CAM 已儲存: {cam_path}")
        
        # === 預測結果 ===
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
        
        pred_resized = pred[:original_size[0], :original_size[1]]
        
        # === 組合圖 ===
        self._create_summary_figure(
            image_rgb, cam_resized, pred_resized, 
            output_dir / f"{image_path.stem}_summary.png"
        )
        
        # === Feature Maps ===
        if show_feature_maps:
            logger.info("提取 Feature Maps...")
            feature_maps = self.feature_viz.extract(input_tensor)
            self._visualize_feature_maps(
                feature_maps, n_feature_maps,
                output_dir / f"{image_path.stem}_features.png"
            )
    
    def _resize_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """調整 CAM 尺寸"""
        from skimage.transform import resize
        return resize(cam, target_size, mode='constant', anti_aliasing=True)
    
    def _create_heatmap_overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        建立熱力圖疊加
        
        Args:
            image: 原始影像 (H, W, 3)
            cam: CAM 熱力圖 (H, W)，值域 [0, 1]
            alpha: 疊加透明度
            
        Returns:
            overlay: 疊加圖 (H, W, 3)
        """
        # 使用 jet colormap
        heatmap = cm.jet(cam)[:, :, :3]  # (H, W, 3)
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # 疊加
        image_float = image.astype(np.float32)
        heatmap_float = heatmap.astype(np.float32)
        
        overlay = (1 - alpha) * image_float + alpha * heatmap_float
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
    
    def _create_summary_figure(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        pred: np.ndarray,
        save_path: Path,
    ) -> None:
        """建立摘要圖"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 原圖
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Grad-CAM 熱力圖
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM (Membrane Attention)', fontsize=14)
        axes[1].axis('off')
        
        # 疊加圖
        overlay = self._create_heatmap_overlay(image, cam, alpha=0.4)
        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Overlay', fontsize=14)
        axes[2].axis('off')
        
        # 預測結果
        axes[3].imshow(pred, cmap='gray')
        axes[3].set_title('Prediction (White=Membrane)', fontsize=14)
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"摘要圖已儲存: {save_path}")
    
    def _visualize_feature_maps(
        self,
        feature_maps: Dict[str, np.ndarray],
        n_layers: int,
        save_path: Path,
    ) -> None:
        """視覺化特徵圖"""
        # 取前 n 層
        layers = list(feature_maps.items())[:n_layers]
        
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
        if len(layers) == 1:
            axes = [axes]
        
        for ax, (name, fmap) in zip(axes, layers):
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'{name}\nShape: {fmap.shape}', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature Maps 已儲存: {save_path}")
    
    def visualize_first_conv_filters(self, save_path: Path) -> None:
        """
        視覺化第一層卷積核
        
        這能顯示模型學到的基礎顏色/邊緣探測器
        """
        # 取得第一層卷積權重
        first_conv = None
        for module in self.model.encoder.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is None:
            logger.warning("找不到第一層卷積")
            return
        
        weights = first_conv.weight.data.cpu().numpy()
        n_filters = min(32, weights.shape[0])
        
        # 正規化權重以便視覺化
        weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())
        
        # 繪製
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
        axes = axes.flatten()
        
        for i in range(n_filters):
            # 取前 3 個 channel (RGB)
            kernel = weights_normalized[i, :3, :, :]
            kernel = np.transpose(kernel, (1, 2, 0))  # (H, W, 3)
            
            axes[i].imshow(kernel)
            axes[i].set_title(f'Filter {i}', fontsize=8)
            axes[i].axis('off')
        
        for i in range(n_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('First Conv Layer Filters (Color/Edge Detectors)', fontsize=14)
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"第一層卷積核已儲存: {save_path}")


def main() -> None:
    """主程式入口"""
    config = load_config()
    
    # 設定路徑
    model_path = config.model_save_dir / "best_model.pth"
    input_dir = config.base_dir / "train" / "test"
    output_dir = config.base_dir / "output" / "watch_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("UNet++ 模型可解釋性分析")
    logger.info("=" * 60)
    
    # 建立觀察器
    watcher = UNetWatcher(
        model_path=model_path,
        encoder_name=config.encoder_name,
        num_classes=config.num_classes,
        image_size=config.image_size,
    )
    
    # 視覺化第一層卷積核
    watcher.visualize_first_conv_filters(output_dir / "first_conv_filters.png")
    
    # 收集影像
    extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
    image_paths = sorted(set(image_paths))
    
    # 隨機抽樣 5 張
    import random
    if len(image_paths) > 5:
        image_paths = random.sample(image_paths, 5)
        logger.info("從測試集中隨機選取 5 張影像")
    
    logger.info(f"找到 {len(image_paths)} 張影像")
    
    # 分析每張影像
    for img_path in image_paths:
        logger.info(f"\n分析: {img_path.name}")
        watcher.analyze(
            image_path=img_path,
            output_dir=output_dir,
            show_feature_maps=True,
            n_feature_maps=4,
        )
    
    logger.info("=" * 60)
    logger.info(f"分析完成！結果已儲存至: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
