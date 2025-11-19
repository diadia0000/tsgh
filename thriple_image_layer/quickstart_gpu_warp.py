"""
Quick Start Example for GPU Warp Engine

This script demonstrates the simplest way to use the GPU Warp Engine.
Run this after completing Module 2 (alignment).

Usage:
    python quickstart_gpu_warp.py

Author: AI Assistant
Date: 2025-05-19
"""

from pathlib import Path
import torch
from gpu_warp_engine import GPUWarpEngine


def main():
    """Quick start example - warp a single slide at Level 2."""

    print("\n" + "="*70)
    print("GPU WARP ENGINE - QUICK START")
    print("="*70 + "\n")

    # ==================== CONFIGURATION ====================
    # Modify these paths to match your setup

    registrar_path = Path(r"H:\tsgh\thriple_image_layer\output\Transform_Params\data\Transform_Params_registrar.pickle")
    slide_name = "DISH_40X_2.czi"  # Change this to your slide name
    output_dir = Path(r"H:\tsgh\thriple_image_layer\output")

    # Processing parameters
    level = 2          # Pyramid level (0=full res, 1=half, 2=quarter, etc.)
    tile_size = 2048   # Tile size for processing

    # =======================================================

    # Verify registrar exists
    if not registrar_path.exists():
        print(f"❌ Error: Registrar not found at {registrar_path}")
        print("\n📝 Please run Module 2 (alignment) first:")
        print("   python thriple_image_layer/run_full_pipeline.py")
        return

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU Memory: {gpu_mem:.1f} GB\n")
        device = 'cuda'
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")
        print("   For GPU acceleration, install CUDA and PyTorch with CUDA support\n")
        device = 'cpu'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Configuration:")
    print(f"   Slide: {slide_name}")
    print(f"   Level: {level} (resolution = 1/{2**level})")
    print(f"   Tile size: {tile_size}x{tile_size}")
    print(f"   Device: {device}")
    print()

    # ==================== PROCESSING ====================

    try:
        print("🚀 Initializing GPU Warp Engine...")
        engine = GPUWarpEngine(
            registrar_path=registrar_path,
            slide_name=slide_name,
            device=device,
            use_non_rigid=True
        )

        # Output path
        output_path = output_dir / f"{Path(slide_name).stem}_warped_lv{level}.tiff"

        print(f"\n🔄 Warping slide...")
        print(f"   This may take several minutes depending on image size...\n")

        engine.warp_full_slide(
            output_path=output_path,
            level=level,
            tile_size=tile_size,
            compression='jpeg',
            quality=90
        )

        print(f"\n✅ SUCCESS!")
        print(f"   Warped image saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024**2:.1f} MB")

        # Suggestions for next steps
        print(f"\n📝 Next steps:")
        print(f"   1. View the output: {output_path}")
        print(f"   2. Try Level 0 for full resolution (may take longer)")
        print(f"   3. Process additional slides by changing 'slide_name'")
        print(f"   4. Run test suite: python test_gpu_warp_engine.py")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n📝 Troubleshooting:")
        print(f"   1. Verify the slide file exists in the CZI directory")
        print(f"   2. Check that the slide name matches the registrar")
        print(f"   3. Run alignment (Module 2) if not done yet")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"\n📝 For detailed debugging:")
        print(f"   - Run with Python debugger: python -m pdb quickstart_gpu_warp.py")
        print(f"   - Check GPU memory: nvidia-smi")
        print(f"   - Review error log above")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

