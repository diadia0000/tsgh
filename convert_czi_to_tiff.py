#!/usr/bin/env python3
"""
Stage 1: CZI to TIFF Conversion
Converts CZI files from picture/ directory to TIFF format in picture/tiff/
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

try:
    from czifile import CziFile
except ImportError:
    print("Error: czifile package not found. Install with: pip install czifile")
    sys.exit(1)

def convert_czi_to_tiff(czi_path, output_path):
    """Convert single CZI file to TIFF format"""
    try:
        with CziFile(czi_path) as czi:
            # Read image data
            image_data = czi.asarray()
            
            # Handle different CZI structures
            if image_data.ndim > 3:
                # Take first channel if multi-dimensional
                while image_data.ndim > 3:
                    image_data = image_data[0]
            
            # Convert to RGB if needed
            if image_data.shape[-1] == 1:
                image_data = np.repeat(image_data, 3, axis=-1)
            elif image_data.shape[-1] > 3:
                image_data = image_data[:, :, :3]
            
            # Normalize to 8-bit
            if image_data.dtype != np.uint8:
                image_data = ((image_data - image_data.min()) / 
                             (image_data.max() - image_data.min()) * 255).astype(np.uint8)
            
            # Save as TIFF
            cv2.imwrite(str(output_path), cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
            return True
            
    except Exception as e:
        print(f"Error converting {czi_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert CZI files to TIFF format")
    parser.add_argument("--input", default="picture/", help="Input directory containing CZI files")
    parser.add_argument("--output", default="picture/tiff/", help="Output directory for TIFF files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CZI files
    czi_files = list(input_dir.glob("*.czi"))
    
    if not czi_files:
        print(f"No CZI files found in {input_dir}")
        return 1
    
    print(f"Found {len(czi_files)} CZI files")
    
    success_count = 0
    for czi_file in czi_files:
        output_file = output_dir / (czi_file.stem + ".tiff")
        print(f"Converting {czi_file.name} -> {output_file.name}")
        
        if convert_czi_to_tiff(czi_file, output_file):
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed")
    
    print(f"\nConversion complete: {success_count}/{len(czi_files)} files converted successfully")
    
    if success_count == len(czi_files):
        print("All files converted successfully!")
        return 0
    else:
        print("Some files failed to convert")
        return 1

if __name__ == "__main__":
    sys.exit(main())