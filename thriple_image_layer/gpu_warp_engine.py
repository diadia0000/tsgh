"""
GPU-Based Inverse Mapping Warp Engine for Gigapixel Pathology Images

This module implements a high-performance inverse mapping pipeline that manually
warps image tiles using PyTorch GPU operations, avoiding memory issues with
valis's built-in warping at full resolution.

Key Features:
- Tile-based processing for gigapixel images
- GPU-accelerated inverse mapping with PyTorch
- Smart dynamic source cropping to minimize I/O
- Support for both rigid and non-rigid transformations
- Compatible with valis registrar output and aicspylibczi readers

Author: AI Assistant
Date: 2025-05-19
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
import pyvips
from aicspylibczi import CziFile
import tifffile
from tqdm import tqdm


class GPUWarpEngine:
    """
    GPU-accelerated inverse mapping engine for warping pathology image tiles.

    This class loads registration parameters from a valis registrar and performs
    inverse mapping to warp tiles from source CZI images to aligned target space.
    """

    def __init__(
        self,
        registrar_path: Path,
        slide_name: str,
        device: str = 'cuda',
        use_non_rigid: bool = True
    ):
        """
        Initialize the GPU warp engine.

        Args:
            registrar_path: Path to the valis registrar pickle file
            slide_name: Name of the slide to warp (e.g., 'DISH_40X_2.czi')
            device: PyTorch device ('cuda' or 'cpu')
            use_non_rigid: Whether to apply non-rigid transformation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.slide_name = slide_name
        self.use_non_rigid = use_non_rigid

        print(f"Initializing GPUWarpEngine on device: {self.device}")

        # Load registrar
        self.registrar = self._load_registrar(registrar_path)

        # Get slide object from registrar
        if slide_name not in self.registrar.slide_dict:
            raise ValueError(f"Slide '{slide_name}' not found in registrar. "
                           f"Available slides: {list(self.registrar.slide_dict.keys())}")

        self.slide_obj = self.registrar.slide_dict[slide_name]

        # Load transformation parameters
        self._load_transforms()

        # Initialize CZI reader
        self.czi_reader = CziFile(self.slide_obj.src_f)
        self.src_shape = self._get_source_shape()

        print(f"✓ Engine initialized for '{slide_name}'")
        print(f"  - Source shape (Level 0): {self.src_shape}")
        print(f"  - Target shape: {(self.slide_obj.processed_img_shape_rc)}")
        print(f"  - Non-rigid: {self.use_non_rigid}")

    def _load_registrar(self, registrar_path: Path):
        """Load the valis registrar from pickle file."""
        with open(registrar_path, 'rb') as f:
            registrar = pickle.load(f)
        print(f"✓ Loaded registrar from {registrar_path}")
        return registrar

    def _load_transforms(self):
        """Load and prepare transformation matrices and displacement fields."""
        # Inverse rigid transformation matrix (3x3)
        self.inv_rigid_matrix = torch.from_numpy(
            self.slide_obj.M  # valis stores the forward matrix in M
        ).float().to(self.device)

        # Invert the matrix for inverse mapping
        self.inv_rigid_matrix = torch.inverse(self.inv_rigid_matrix)

        # Load non-rigid displacement fields if available
        if self.use_non_rigid and hasattr(self.slide_obj, 'bk_dxdy'):
            # bk_dxdy contains backward (inverse) displacement fields
            dxdy = self.slide_obj.bk_dxdy  # This is a pyvips image

            # Convert pyvips to numpy
            if isinstance(dxdy, pyvips.Image):
                # Get displacement field shape
                self.dvf_shape = (dxdy.height, dxdy.width)

                # Convert to numpy array (channels: dx, dy)
                dx_np = np.ndarray(
                    buffer=dxdy.extract_band(0).write_to_memory(),
                    dtype=np.float32,
                    shape=(dxdy.height, dxdy.width)
                )
                dy_np = np.ndarray(
                    buffer=dxdy.extract_band(1).write_to_memory(),
                    dtype=np.float32,
                    shape=(dxdy.height, dxdy.width)
                )

                # Stack and convert to torch tensor (1, H, W, 2)
                dxdy_np = np.stack([dx_np, dy_np], axis=-1)
                self.dvf = torch.from_numpy(dxdy_np).unsqueeze(0).to(self.device)

                print(f"✓ Loaded non-rigid DVF: {self.dvf_shape}")
            else:
                print("⚠ No valid displacement field found, using rigid-only")
                self.dvf = None
                self.dvf_shape = None
        else:
            self.dvf = None
            self.dvf_shape = None
            print("✓ Using rigid-only transformation")

    def _get_source_shape(self) -> Tuple[int, int]:
        """Get the shape of the source image at Level 0."""
        # Get dimensions from CZI reader
        # Use the slide object's processed shape as reference
        # The source shape should be in slide_obj attributes
        if hasattr(self.slide_obj, 'original_img_shape_rc'):
            height, width = self.slide_obj.original_img_shape_rc[:2]
        else:
            # Fallback: read a small region to get dimensions
            # CziFile provides size through its metadata
            try:
                # Read metadata to get dimensions
                meta = self.czi_reader.get_mosaic_bounding_box()
                width = meta.w
                height = meta.h
            except:
                # Final fallback: use processed shape
                height, width = self.slide_obj.processed_img_shape_rc[:2]
        return (height, width)

    def process_tile(
        self,
        level: int,
        tile_x: int,
        tile_y: int,
        tile_w: int,
        tile_h: int,
        output_channels: int = 3
    ) -> np.ndarray:
        """
        Process a single tile using inverse mapping.

        This is the core method that implements the inverse mapping pipeline:
        1. Generate target grid coordinates
        2. Apply inverse rigid transform
        3. Apply inverse non-rigid transform
        4. Smart crop from source CZI
        5. GPU-based pixel sampling

        Args:
            level: Pyramid level (0 = full resolution)
            tile_x: X coordinate of tile in target space (at given level)
            tile_y: Y coordinate of tile in target space (at given level)
            tile_w: Width of tile
            tile_h: Height of tile
            output_channels: Number of output channels (3 for RGB)

        Returns:
            Warped tile as numpy array (H, W, C) uint8
        """
        # Scale factor for pyramid level
        scale_factor = 2 ** level

        # Convert to Level 0 coordinates
        x0 = tile_x * scale_factor
        y0 = tile_y * scale_factor
        w0 = tile_w * scale_factor
        h0 = tile_h * scale_factor

        # Step 1: Generate target grid (GPU)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h0, dtype=torch.float32, device=self.device),
            torch.arange(w0, dtype=torch.float32, device=self.device),
            indexing='ij'
        )

        # Offset by tile position to get global coordinates
        grid_x = grid_x + x0
        grid_y = grid_y + y0

        # Step 2: Apply inverse rigid transform
        x_rigid, y_rigid = self._apply_rigid_transform(grid_x, grid_y)

        # Step 3: Apply inverse non-rigid transform (if available)
        if self.dvf is not None:
            x_src, y_src = self._apply_nonrigid_transform(x_rigid, y_rigid)
        else:
            x_src, y_src = x_rigid, y_rigid

        # Step 4: Dynamic source cropping (the "smart read")
        source_img, bbox_info = self._crop_source_region(x_src, y_src, level)

        if source_img is None:
            # Return black tile if completely out of bounds
            return np.zeros((tile_h, tile_w, output_channels), dtype=np.uint8)

        # Step 5: Final pixel sampling with grid_sample
        warped_tile = self._sample_pixels(source_img, x_src, y_src, bbox_info)

        # Downsample if needed
        if scale_factor > 1:
            warped_tile = F.interpolate(
                warped_tile,
                size=(tile_h, tile_w),
                mode='bilinear',
                align_corners=False
            )

        # Convert to numpy (C, H, W) -> (H, W, C)
        result = warped_tile.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _apply_rigid_transform(
        self,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse rigid transformation to coordinate grid.

        Args:
            grid_x: X coordinates in target space (H, W)
            grid_y: Y coordinates in target space (H, W)

        Returns:
            Transformed x and y coordinates
        """
        # Create homogeneous coordinates (H, W, 3)
        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)

        # Apply transformation: [x', y', 1]^T = M^-1 @ [x, y, 1]^T
        # coords: (H, W, 3), matrix: (3, 3)
        transformed = torch.matmul(coords, self.inv_rigid_matrix.T)  # (H, W, 3)

        x_rigid = transformed[..., 0]
        y_rigid = transformed[..., 1]

        return x_rigid, y_rigid

    def _apply_nonrigid_transform(
        self,
        x_rigid: torch.Tensor,
        y_rigid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse non-rigid displacement field.

        Args:
            x_rigid: X coordinates after rigid transform (H, W)
            y_rigid: Y coordinates after rigid transform (H, W)

        Returns:
            Final source coordinates with non-rigid deformation
        """
        H, W = x_rigid.shape

        # Normalize coordinates to DVF space [-1, 1]
        dvf_h, dvf_w = self.dvf_shape

        # Normalize to [-1, 1] for grid_sample
        x_norm = 2.0 * x_rigid / (self.src_shape[1] - 1) - 1.0
        y_norm = 2.0 * y_rigid / (self.src_shape[0] - 1) - 1.0

        # Stack to grid (1, H, W, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        # Sample displacement field at these locations
        # dvf: (1, H_dvf, W_dvf, 2), grid: (1, H, W, 2)
        displacement = F.grid_sample(
            self.dvf.permute(0, 3, 1, 2),  # (1, 2, H_dvf, W_dvf)
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # (1, 2, H, W)

        # Convert back to (H, W) per channel
        dx = displacement[0, 0, :, :]
        dy = displacement[0, 1, :, :]

        # Add displacement to get final source coordinates
        x_src = x_rigid + dx
        y_src = y_rigid + dy

        return x_src, y_src

    def _crop_source_region(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        level: int
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """
        Dynamically crop the source region from CZI based on coordinate bounds.

        Args:
            x_src: Source X coordinates (H, W)
            y_src: Source Y coordinates (H, W)
            level: Pyramid level for reading

        Returns:
            Tuple of (source_image_tensor, bbox_info)
            - source_image_tensor: (1, C, H_crop, W_crop) on GPU
            - bbox_info: Dict with crop bounds and padding info
        """
        # Calculate bounding box
        x_min = torch.floor(x_src.min()).int().item()
        x_max = torch.ceil(x_src.max()).int().item()
        y_min = torch.floor(y_src.min()).int().item()
        y_max = torch.ceil(y_src.max()).int().item()

        # Add small padding for interpolation
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(self.src_shape[1] - 1, x_max + padding)
        y_max = min(self.src_shape[0] - 1, y_max + padding)

        crop_w = x_max - x_min + 1
        crop_h = y_max - y_min + 1

        # Check if completely out of bounds
        if crop_w <= 0 or crop_h <= 0:
            return None, None

        # Read from CZI using aicspylibczi
        try:
            # CZI uses (x, y, w, h) format
            # Read at specified level (pyramid level)
            img_data = self.czi_reader.read_mosaic(
                region=(x_min, y_min, crop_w, crop_h),
                scale_factor=1.0 / (2 ** level),  # Scale for pyramid level
                C=0  # Read all channels
            )

            # img_data shape: (1, 1, 1, H, W, C) or similar
            # Squeeze unnecessary dimensions
            while img_data.ndim > 3:
                if img_data.shape[0] == 1:
                    img_data = img_data.squeeze(0)
                elif img_data.shape[-1] <= 4:  # Channels at end
                    break
                else:
                    break

            # Ensure (H, W, C) format
            if img_data.ndim == 2:
                img_data = img_data[:, :, np.newaxis]

            # Convert to torch tensor (C, H, W)
            img_tensor = torch.from_numpy(img_data).float()
            if img_tensor.ndim == 3 and img_tensor.shape[-1] <= 4:
                img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

            img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

            # Store bbox info for coordinate normalization
            bbox_info = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'crop_w': crop_w,
                'crop_h': crop_h
            }

            return img_tensor, bbox_info

        except Exception as e:
            print(f"⚠ Error reading CZI region: {e}")
            return None, None

    def _sample_pixels(
        self,
        source_img: torch.Tensor,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        bbox_info: Dict
    ) -> torch.Tensor:
        """
        Sample pixels from source image using grid_sample.

        Args:
            source_img: Source image tensor (1, C, H_crop, W_crop)
            x_src: Source X coordinates in global space (H, W)
            y_src: Source Y coordinates in global space (H, W)
            bbox_info: Bounding box information

        Returns:
            Sampled tensor (1, C, H, W)
        """
        # Normalize coordinates to [-1, 1] relative to cropped region
        x_min = bbox_info['x_min']
        y_min = bbox_info['y_min']
        crop_w = bbox_info['crop_w']
        crop_h = bbox_info['crop_h']

        # Convert global coordinates to local crop coordinates
        x_local = x_src - x_min
        y_local = y_src - y_min

        # Normalize to [-1, 1] for grid_sample
        x_norm = 2.0 * x_local / (crop_w - 1) - 1.0
        y_norm = 2.0 * y_local / (crop_h - 1) - 1.0

        # Stack to grid (1, H, W, 2)
        # grid_sample expects (x, y) order
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        # Sample pixels
        sampled = F.grid_sample(
            source_img,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return sampled

    def warp_full_slide(
        self,
        output_path: Path,
        level: int = 0,
        tile_size: int = 2048,
        compression: str = 'jpeg',
        quality: int = 90
    ):
        """
        Warp the entire slide tile-by-tile and save to BigTIFF.

        Args:
            output_path: Output TIFF file path
            level: Pyramid level to process
            tile_size: Size of processing tiles
            compression: TIFF compression ('jpeg', 'lzw', 'deflate', None)
            quality: JPEG quality (if using jpeg compression)
        """
        # Get target dimensions
        scale_factor = 2 ** level
        target_shape = self.slide_obj.processed_img_shape_rc
        target_h = target_shape[0] // scale_factor
        target_w = target_shape[1] // scale_factor

        print(f"\n{'='*60}")
        print(f"Warping full slide: {self.slide_name}")
        print(f"  - Target size: {target_h} x {target_w} (Level {level})")
        print(f"  - Tile size: {tile_size}")
        print(f"  - Output: {output_path}")
        print(f"{'='*60}\n")

        # Calculate tile grid
        n_tiles_x = (target_w + tile_size - 1) // tile_size
        n_tiles_y = (target_h + tile_size - 1) // tile_size
        total_tiles = n_tiles_x * n_tiles_y

        print(f"Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles\n")

        # Initialize output array (memory-mapped for large images)
        # For very large images, consider using tifffile with tile writing
        output_shape = (target_h, target_w, 3)

        # Process tiles with progress bar
        with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
            tiles_data = []

            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    # Calculate tile coordinates
                    x = tx * tile_size
                    y = ty * tile_size
                    w = min(tile_size, target_w - x)
                    h = min(tile_size, target_h - y)

                    # Process tile
                    tile = self.process_tile(level, x, y, w, h)

                    # Pad if necessary
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded[:tile.shape[0], :tile.shape[1], :] = tile
                        tile = padded

                    tiles_data.append((y, x, tile))
                    pbar.update(1)

        # Assemble tiles and write to TIFF
        print("\nAssembling and writing output...")
        output_img = np.zeros(output_shape, dtype=np.uint8)

        for y, x, tile in tiles_data:
            h, w = tile.shape[:2]
            output_img[y:y+h, x:x+w, :] = tile[:h, :w, :]

        # Write to TIFF
        tifffile.imwrite(
            output_path,
            output_img,
            compression=compression,
            compressionargs={'level': quality} if compression == 'jpeg' else None,
            photometric='rgb',
            tile=(tile_size, tile_size),
            metadata={'axes': 'YXC'}
        )

        print(f"✓ Output saved to: {output_path}")
        print(f"  - Final size: {output_img.shape}")


def main():
    """
    Demonstration of GPUWarpEngine usage.

    This example shows how to:
    1. Initialize the engine with a registrar
    2. Process a slide tile-by-tile
    3. Save the result as a BigTIFF
    """
    # Configuration
    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output\gpu_warped")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Slides to process
    slides_to_warp = [
        "DISH_40X_2.czi",
        "HER2_40X.czi"
    ]

    # Processing parameters
    level = 2  # Process at Level 2 (1/4 resolution)
    tile_size = 2048  # 2K tiles for good GPU utilization

    print("\n" + "="*60)
    print("GPU Warp Engine - Batch Processing")
    print("="*60 + "\n")

    # Process each slide
    for slide_name in slides_to_warp:
        try:
            # Initialize engine
            engine = GPUWarpEngine(
                registrar_path=registrar_path,
                slide_name=slide_name,
                device='cuda',
                use_non_rigid=True
            )

            # Output path
            output_path = output_dir / f"{Path(slide_name).stem}_warped_lv{level}.tiff"

            # Warp full slide
            engine.warp_full_slide(
                output_path=output_path,
                level=level,
                tile_size=tile_size,
                compression='jpeg',
                quality=90
            )

            print(f"\n✓ Successfully processed: {slide_name}\n")

        except Exception as e:
            print(f"\n✗ Failed to process {slide_name}: {e}\n")
            import traceback
            traceback.print_exc()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("Batch processing complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

