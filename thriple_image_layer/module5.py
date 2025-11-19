"""
Module 5: GPU-Accelerated Full Resolution Image Warping

This module uses the custom GPUWarpEngine to warp registered images at full
resolution (Level 0) or any specified pyramid level, processing tiles
on-the-fly to avoid memory issues.

Key Features:
- Tile-based processing for gigapixel images
- GPU acceleration with PyTorch
- Support for multi-channel merging (DISH + HER2)
- Efficient inverse mapping without loading full images

Author: AI Assistant
Date: 2025-05-19
"""

from pathlib import Path

from typing import List, Optional
import numpy as np
import torch
import tifffile
from tqdm import tqdm

from gpu_warp_engine import GPUWarpEngine


def warp_and_merge_slides(
    registrar_path: Path,
    output_dir: Path,
    slides_to_warp: List[str],
    level: int = 2,
    tile_size: int = 2048,
    merge: bool = True,
    compression: str = 'jpeg',
    quality: int = 90
):
    """
    Warp multiple slides and optionally merge them into a single RGB image.

    Args:
        registrar_path: Path to valis registrar pickle file
        output_dir: Output directory for warped images
        slides_to_warp: List of slide names to warp (e.g., ['DISH_40X_2.czi', 'HER2_40X.czi'])
        level: Pyramid level to process (0=full resolution, 1=half, 2=quarter, etc.)
        tile_size: Size of tiles for processing
        merge: Whether to merge warped slides into a single RGB image
        compression: TIFF compression method
        quality: Compression quality
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MODULE 5: GPU-ACCELERATED IMAGE WARPING")
    print("="*70)
    print(f"  Registrar: {registrar_path}")
    print(f"  Slides to warp: {slides_to_warp}")
    print(f"  Output level: {level}")
    print(f"  Tile size: {tile_size}")
    print(f"  Merge slides: {merge}")
    print("="*70 + "\n")

    warped_images = {}

    # Step 1: Warp each slide individually
    for slide_name in slides_to_warp:
        print(f"\n{'─'*70}")
        print(f"Processing: {slide_name}")
        print(f"{'─'*70}\n")

        try:
            # Initialize GPU engine
            engine = GPUWarpEngine(
                registrar_path=registrar_path,
                slide_name=slide_name,
                device='cuda',
                use_non_rigid=True
            )

            # Output path for individual warped image
            output_path = output_dir / f"{Path(slide_name).stem}_warped_lv{level}.tiff"

            # Warp the slide
            engine.warp_full_slide(
                output_path=output_path,
                level=level,
                tile_size=tile_size,
                compression=compression,
                quality=quality
            )

            # Store warped image path for tile-by-tile merging
            if merge:
                warped_images[slide_name] = output_path  # ✅ Store path instead of loading full image

            print(f"\n✓ Successfully warped: {slide_name}")

        except Exception as e:
            print(f"\n✗ Error warping {slide_name}: {e}")
            import traceback
            traceback.print_exc()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 2: Merge warped images if requested (tile-by-tile to avoid OOM)
    if merge and len(warped_images) > 0:
        print(f"\n{'─'*70}")
        print("Merging warped images (tile-by-tile)...")
        print(f"{'─'*70}\n")

        merge_output = output_dir / f"Merged_Aligned_lv{level}.tiff"

        # Use tile-by-tile merge to avoid loading full images
        merge_channels_tiled(
            warped_paths=warped_images,
            output_path=merge_output,
            slide_names=slides_to_warp,
            tile_size=tile_size,
            compression=compression,
            quality=quality
        )

        print(f"✓ Merged image saved: {merge_output}")

    print("\n" + "="*70)
    print("MODULE 5 COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70 + "\n")


def merge_channels_tiled(
    warped_paths: dict,
    output_path: Path,
    slide_names: List[str],
    tile_size: int = 2048,
    channel_mapping: Optional[dict] = None,
    compression: str = 'jpeg',
    quality: int = 90
) -> None:
    """
    Merge multiple warped images into a single RGB image using tile-by-tile processing.

    This avoids loading full gigapixel images into memory, making it suitable for
    Level 0 (full resolution) processing.

    Args:
        warped_paths: Dictionary of {slide_name: Path to warped TIFF}
        output_path: Output path for merged image
        slide_names: Ordered list of slide names
        tile_size: Size of tiles for processing
        channel_mapping: Optional mapping of slide names to RGB channels
                        Default: First image -> R, Second -> G, Third -> B
        compression: TIFF compression method
        quality: Compression quality for JPEG
    """
    if not warped_paths:
        raise ValueError("No warped images to merge")

    # Default channel mapping: assign each image to a different channel
    if channel_mapping is None:
        channel_mapping = {}
        for i, name in enumerate(slide_names[:3]):  # Max 3 channels for RGB
            channel_mapping[name] = i

    print(f"Merging {len(warped_paths)} images (tile-by-tile):")
    for slide_name in warped_paths.keys():
        if slide_name in channel_mapping:
            channel = channel_mapping[slide_name]
            print(f"  - {slide_name} -> Channel {channel} ({'RGB'[channel]})")

    # Open all input files (read metadata only)
    readers = {}
    for slide_name, path in warped_paths.items():
        if slide_name in channel_mapping:
            readers[slide_name] = tifffile.TiffFile(path)

    # Get dimensions from first image
    first_reader = list(readers.values())[0]
    first_page = first_reader.pages[0]
    H, W = first_page.shape[:2]

    print(f"  - Output dimensions: {H} x {W}")
    print(f"  - Tile size: {tile_size}")

    # Calculate tile grid
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    total_tiles = n_tiles_x * n_tiles_y

    print(f"  - Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles\n")

    # Process tiles with progress bar
    merged_tiles = []

    with tqdm(total=total_tiles, desc="Merging tiles") as pbar:
        for ty in range(n_tiles_y):
            row_tiles = []
            for tx in range(n_tiles_x):
                # Calculate tile coordinates
                x = tx * tile_size
                y = ty * tile_size
                tw = min(tile_size, W - x)
                th = min(tile_size, H - y)

                # Initialize output tile
                merged_tile = np.zeros((th, tw, 3), dtype=np.uint8)

                # Read from each image and write to corresponding channel
                for slide_name, reader in readers.items():
                    if slide_name not in channel_mapping:
                        continue

                    channel = channel_mapping[slide_name]

                    # Read tile from this image
                    tile = reader.pages[0].asarray()[y:y+th, x:x+tw]

                    # Convert to grayscale if needed
                    if tile.ndim == 3 and tile.shape[2] >= 3:
                        # Average RGB channels
                        gray = np.mean(tile[:, :, :3], axis=2).astype(np.uint8)
                    elif tile.ndim == 3:
                        gray = tile[:, :, 0]
                    else:
                        gray = tile

                    # Write to output channel
                    merged_tile[:, :, channel] = gray

                row_tiles.append(merged_tile)
                pbar.update(1)

            # Concatenate row
            row_img = np.concatenate(row_tiles, axis=1)
            merged_tiles.append(row_img)

    # Concatenate all rows
    print("\nAssembling final image...")
    merged_img = np.concatenate(merged_tiles, axis=0)

    # Write output
    print(f"Writing output to {output_path}...")
    tifffile.imwrite(
        output_path,
        merged_img,
        compression=compression,
        compressionargs={'level': quality} if compression == 'jpeg' else None,
        photometric='rgb',
        tile=(tile_size, tile_size),
        metadata={'axes': 'YXC'}
    )

    # Close all readers
    for reader in readers.values():
        reader.close()

    print(f"✓ Merge complete: {merged_img.shape}")


def merge_channels(
    warped_images: dict,
    slide_names: List[str],
    channel_mapping: Optional[dict] = None
) -> np.ndarray:
    """
    Merge multiple warped images into a single RGB image (in-memory version).

    ⚠️ WARNING: This function loads entire images into memory.
    For large images (Level 0-1), use merge_channels_tiled instead.

    Args:
        warped_images: Dictionary of {slide_name: warped_image_array}
        slide_names: Ordered list of slide names
        channel_mapping: Optional mapping of slide names to RGB channels
                        Default: First image -> R, Second -> G, Third -> B

    Returns:
        Merged RGB image as numpy array (H, W, 3)
    """
    if not warped_images:
        raise ValueError("No warped images to merge")

    # Get output shape from first image
    first_img = list(warped_images.values())[0]
    output_shape = first_img.shape[:2] + (3,)
    merged = np.zeros(output_shape, dtype=np.uint8)

    # Default channel mapping: assign each image to a different channel
    if channel_mapping is None:
        channel_mapping = {}
        for i, name in enumerate(slide_names[:3]):  # Max 3 channels for RGB
            channel_mapping[name] = i

    print(f"Merging {len(warped_images)} images:")
    for slide_name, img in warped_images.items():
        if slide_name not in channel_mapping:
            continue

        channel = channel_mapping[slide_name]

        # Convert to grayscale if needed
        if img.ndim == 3 and img.shape[2] >= 3:
            # Take first channel or convert to grayscale
            gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
        elif img.ndim == 3:
            gray = img[:, :, 0]
        else:
            gray = img

        merged[:, :, channel] = gray
        print(f"  - {slide_name} -> Channel {channel} ({'RGB'[channel]})")

    return merged


def warp_single_tile_demo(
    registrar_path: Path,
    slide_name: str,
    tile_x: int = 0,
    tile_y: int = 0,
    tile_size: int = 2048,
    level: int = 2,
    output_path: Optional[Path] = None
):
    """
    Demonstration function to warp a single tile and optionally save it.

    This is useful for testing and debugging the GPU warp engine.

    Args:
        registrar_path: Path to registrar pickle
        slide_name: Name of slide to warp
        tile_x: X coordinate of tile
        tile_y: Y coordinate of tile
        tile_size: Size of tile
        level: Pyramid level
        output_path: Optional path to save tile

    Returns:
        Warped tile as numpy array
    """
    print(f"\n{'='*60}")
    print(f"Single Tile Warp Demo")
    print(f"{'='*60}")
    print(f"  Slide: {slide_name}")
    print(f"  Tile: ({tile_x}, {tile_y}, {tile_size}x{tile_size})")
    print(f"  Level: {level}")
    print(f"{'='*60}\n")

    # Initialize engine
    engine = GPUWarpEngine(
        registrar_path=registrar_path,
        slide_name=slide_name,
        device='cuda',
        use_non_rigid=True
    )

    # Process tile
    print(f"Processing tile...")
    tile = engine.process_tile(level, tile_x, tile_y, tile_size, tile_size)

    print(f"✓ Tile processed: shape={tile.shape}, dtype={tile.dtype}")

    # Save if requested
    if output_path:
        tifffile.imwrite(output_path, tile, photometric='rgb')
        print(f"✓ Tile saved to: {output_path}")

    return tile


def main():
    """
    Main execution function for Module 5.

    This processes the standard pipeline slides (HER2, DISH, HE) and generates
    aligned full-resolution images.
    """
    # Paths
    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output")

    # Verify registrar exists
    if not registrar_path.exists():
        print(f"✗ Error: Registrar not found at {registrar_path}")
        print("  Please run Module 2 (alignment) first.")
        return

    # Configuration
    slides_to_warp = [
        "HER2_40X.czi",   # Reference (usually not warped, but included)
        "DISH_40X_2.czi"  # Needs warping
    ]

    # Processing parameters
    level = 2  # Start with Level 2 for faster processing
    tile_size = 2048

    # Run warping
    warp_and_merge_slides(
        registrar_path=registrar_path,
        output_dir=output_dir,
        slides_to_warp=slides_to_warp,
        level=level,
        tile_size=tile_size,
        merge=True,
        compression='deflate'
    )

    print("\n✓ Module 5 execution complete!")


if __name__ == "__main__":
    main()

