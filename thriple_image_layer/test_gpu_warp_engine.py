"""
Test Script for GPU Warp Engine

This script provides comprehensive tests for the GPUWarpEngine:
1. Single tile processing test
2. Small region warping test
3. Full slide warping test (configurable levels)
4. Performance benchmarking

Author: AI Assistant
Date: 2025-05-19
"""

from pathlib import Path
import time
import torch
import numpy as np
import tifffile
from gpu_warp_engine import GPUWarpEngine


def test_single_tile():
    """Test 1: Process a single tile and verify output."""
    print("\n" + "="*70)
    print("TEST 1: Single Tile Processing")
    print("="*70 + "\n")

    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output\test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_name = "DISH_40X_2.czi"
    level = 2
    tile_size = 2048

    try:
        # Initialize engine
        engine = GPUWarpEngine(
            registrar_path=registrar_path,
            slide_name=slide_name,
            device='cuda',
            use_non_rigid=True
        )

        # Process a single tile at the origin
        print("Processing tile at (0, 0)...")
        start_time = time.time()

        tile = engine.process_tile(
            level=level,
            tile_x=0,
            tile_y=0,
            tile_w=tile_size,
            tile_h=tile_size
        )

        elapsed = time.time() - start_time

        # Verify output
        print(f"\n✓ Tile processed successfully!")
        print(f"  - Shape: {tile.shape}")
        print(f"  - Dtype: {tile.dtype}")
        print(f"  - Value range: [{tile.min()}, {tile.max()}]")
        print(f"  - Processing time: {elapsed:.3f} seconds")

        # Save tile
        output_path = output_dir / f"test_single_tile_lv{level}.tiff"
        tifffile.imwrite(output_path, tile, photometric='rgb')
        print(f"  - Saved to: {output_path}")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_warping():
    """Test 2: Warp a 2x2 grid of tiles."""
    print("\n" + "="*70)
    print("TEST 2: Grid Warping (2x2 tiles)")
    print("="*70 + "\n")

    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output\test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_name = "DISH_40X_2.czi"
    level = 2
    tile_size = 1024  # Smaller tiles for faster testing

    try:
        # Initialize engine
        engine = GPUWarpEngine(
            registrar_path=registrar_path,
            slide_name=slide_name,
            device='cuda',
            use_non_rigid=True
        )

        # Process 2x2 grid
        print("Processing 2x2 tile grid...")
        start_time = time.time()

        tiles = []
        for ty in range(2):
            row_tiles = []
            for tx in range(2):
                x = tx * tile_size
                y = ty * tile_size

                print(f"  - Processing tile ({tx}, {ty}) at ({x}, {y})...")
                tile = engine.process_tile(level, x, y, tile_size, tile_size)
                row_tiles.append(tile)

            # Concatenate row
            row_img = np.concatenate(row_tiles, axis=1)
            tiles.append(row_img)

        # Concatenate all rows
        full_img = np.concatenate(tiles, axis=0)
        elapsed = time.time() - start_time

        print(f"\n✓ Grid warping complete!")
        print(f"  - Output shape: {full_img.shape}")
        print(f"  - Total time: {elapsed:.3f} seconds")
        print(f"  - Avg time per tile: {elapsed/4:.3f} seconds")

        # Save result
        output_path = output_dir / f"test_grid_2x2_lv{level}.tiff"
        tifffile.imwrite(output_path, full_img, photometric='rgb')
        print(f"  - Saved to: {output_path}")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_slide_warping(level: int = 2):
    """Test 3: Warp a full slide at specified level."""
    print("\n" + "="*70)
    print(f"TEST 3: Full Slide Warping (Level {level})")
    print("="*70 + "\n")

    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output\test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_name = "DISH_40X_2.czi"
    tile_size = 2048

    try:
        # Initialize engine
        engine = GPUWarpEngine(
            registrar_path=registrar_path,
            slide_name=slide_name,
            device='cuda',
            use_non_rigid=True
        )

        # Warp full slide
        output_path = output_dir / f"test_full_slide_lv{level}.tiff"

        start_time = time.time()
        engine.warp_full_slide(
            output_path=output_path,
            level=level,
            tile_size=tile_size,
            compression='jpeg',
            quality=90
        )
        elapsed = time.time() - start_time

        print(f"\n✓ Full slide warping complete!")
        print(f"  - Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"  - Output: {output_path}")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Test 4: Benchmark different tile sizes and GPU utilization."""
    print("\n" + "="*70)
    print("TEST 4: Performance Benchmark")
    print("="*70 + "\n")

    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    slide_name = "DISH_40X_2.czi"
    level = 2

    tile_sizes = [512, 1024, 2048, 4096]
    results = {}

    print("Testing different tile sizes...\n")

    for tile_size in tile_sizes:
        try:
            # Initialize engine
            engine = GPUWarpEngine(
                registrar_path=registrar_path,
                slide_name=slide_name,
                device='cuda',
                use_non_rigid=True
            )

            # Process single tile multiple times and average
            times = []
            n_runs = 3

            print(f"Tile size {tile_size}x{tile_size}:")
            for i in range(n_runs):
                start = time.time()
                tile = engine.process_tile(level, 0, 0, tile_size, tile_size)
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Run {i+1}: {elapsed:.3f}s")

            avg_time = np.mean(times)
            std_time = np.std(times)

            pixels_total = float(tile_size * tile_size)
            results[tile_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'pixels_per_sec': pixels_total / avg_time
            }

            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Throughput: {results[tile_size]['pixels_per_sec']/1e6:.2f} Mpixels/sec\n")

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            continue

    # Print summary
    print("\n" + "─"*70)
    print("BENCHMARK SUMMARY")
    print("─"*70)
    print(f"{'Tile Size':<15} {'Avg Time (s)':<15} {'Throughput (Mpx/s)':<20}")
    print("─"*70)
    for size, data in results.items():
        print(f"{size}x{size:<8} {data['avg_time']:<15.3f} {data['pixels_per_sec']/1e6:<20.2f}")
    print("─"*70 + "\n")

    return True


def test_memory_usage():
    """Test 5: Monitor GPU memory usage during processing."""
    print("\n" + "="*70)
    print("TEST 5: GPU Memory Usage")
    print("="*70 + "\n")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return False

    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    slide_name = "DISH_40X_2.czi"
    level = 2
    tile_size = 2048

    try:
        # Reset GPU memory stats
        torch.cuda.reset_peak_memory_stats()

        initial_mem = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"Initial GPU memory: {initial_mem:.2f} MB")

        # Initialize engine
        engine = GPUWarpEngine(
            registrar_path=registrar_path,
            slide_name=slide_name,
            device='cuda',
            use_non_rigid=True
        )

        after_init_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"After initialization: {after_init_mem:.2f} MB")
        print(f"  (Δ = {after_init_mem - initial_mem:.2f} MB)")

        # Process a tile
        print("\nProcessing tile...")
        tile = engine.process_tile(level, 0, 0, tile_size, tile_size)

        after_process_mem = torch.cuda.memory_allocated() / 1024**2
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"After processing: {after_process_mem:.2f} MB")
        print(f"Peak memory: {peak_mem:.2f} MB")
        print(f"  (Δ from init = {peak_mem - after_init_mem:.2f} MB)")

        # Clean up
        del engine
        torch.cuda.empty_cache()

        final_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"\nAfter cleanup: {final_mem:.2f} MB")

        print(f"\n✓ Memory test complete")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  GPU WARP ENGINE - COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    else:
        print("⚠ CUDA not available, running on CPU (will be slow)\n")

    tests = [
        ("Single Tile Processing", test_single_tile),
        ("Grid Warping", test_grid_warping),
        ("Full Slide Warping (Level 2)", lambda: test_full_slide_warping(level=2)),
        ("Performance Benchmark", test_performance_benchmark),
        ("Memory Usage", test_memory_usage)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✓ PASS" if success else "✗ FAIL"
        except Exception as e:
            results[test_name] = f"✗ ERROR: {e}"

        # Small delay between tests
        time.sleep(1)

    # Print summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST SUMMARY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    for test_name, result in results.items():
        status_symbol = "✓" if "PASS" in result else "✗"
        print(f"{status_symbol} {test_name:<50} {result}")

    print("\n" + "█"*70 + "\n")


if __name__ == "__main__":
    main()

