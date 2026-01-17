import tifffile
import numpy as np
from pathlib import Path

output_dir = Path(r"H:\tsgh\thriple_image_layer\output\lv0_tile")

files = [
    "HER2_40X_warped_lv0.tiff",
    "DISH_40X_2_warped_lv0.tiff",
    "Merged_Aligned_lv0.tiff"
]

print("Checking warped output files...")
print("=" * 70)

for filename in files:
    filepath = output_dir / filename
    if not filepath.exists():
        print(f"\n❌ {filename}: NOT FOUND")
        continue

    # Get file size
    file_size = filepath.stat().st_size

    # Read image
    img = tifffile.imread(filepath)

    # Calculate statistics
    shape = img.shape
    dtype = img.dtype
    min_val = img.min()
    max_val = img.max()
    mean_val = img.mean()
    non_zero = np.count_nonzero(img)
    total = img.size
    percent_nonzero = 100 * non_zero / total

    print(f"\n📄 {filename}")
    print(f"   File size: {file_size / 1024:.2f} KB")
    print(f"   Shape: {shape}, dtype: {dtype}")
    print(f"   Values: min={min_val}, max={max_val}, mean={mean_val:.2f}")
    print(f"   Non-zero pixels: {non_zero:,} / {total:,} ({percent_nonzero:.2f}%)")

    if max_val == 0:
        print(f"   ⚠️  WARNING: Image is completely BLACK!")
    else:
        print(f"   ✅ Image contains data")

print("\n" + "=" * 70)

