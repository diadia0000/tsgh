#!/usr/bin/env python3
"""
Stage 1: CZI to (OME-)TIFF Conversion
Converts CZI files from picture/ directory to TIFF format in picture/tiff/

Enhancements:
- Support writing OME-TIFF with physical pixel size (mpp) metadata
- Allow selecting a specific channel index
- Preserve color without unintended orientation changes
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import warnings
import xml.etree.ElementTree as ET

try:
    import tifffile
except ImportError:  # Lazy guidance for users
    print("Error: tifffile package not found. Install with: pip install tifffile")
    sys.exit(1)

try:
    from czifile import CziFile
except ImportError:
    print("Error: czifile package not found. Install with: pip install czifile")
    sys.exit(1)

def _parse_mpp_from_czi(czi: "CziFile"):
    """Best-effort parse of physical pixel size (µm) from CZI metadata.

    Returns (mpp_x, mpp_y) as floats in microns if available, else (None, None).
    """
    try:
        # czifile exposes XML metadata via .metadata() on some versions or .metadata attribute on others
        try:
            xml_text = czi.metadata()
        except TypeError:
            xml_text = getattr(czi, "metadata", None)
        if not xml_text:
            return None, None
        # Ensure string
        if not isinstance(xml_text, str):
            xml_text = str(xml_text)

        root = ET.fromstring(xml_text)

        mpp_x = None
        mpp_y = None

        # Heuristic search: look for Distance elements with Id X/Y and Value children
        for dist in root.iter():
            tag = dist.tag.split("}")[-1]
            if tag != "Distance":
                continue
            id_val = None
            val = None
            for ch in list(dist):
                ctag = ch.tag.split("}")[-1]
                if ctag == "Id":
                    id_val = (ch.text or "").strip()
                elif ctag == "Value":
                    try:
                        val = float(ch.text)
                    except (TypeError, ValueError):
                        val = None
            if id_val == "X" and val is not None:
                mpp_x = val
            if id_val == "Y" and val is not None:
                mpp_y = val

        return mpp_x, mpp_y
    except Exception:
        return None, None


def _to_yxc(array: np.ndarray) -> np.ndarray:
    """Squeeze and reshape arbitrary CZI array to YXC (channels last).

    Strategy:
    - Squeeze length-1 axes.
    - Identify the two largest axes as Y and X (spatial).
    - Flatten remaining axes into channel dimension.
    """
    arr = np.squeeze(array)
    if arr.ndim == 2:
        return arr[..., None]
    # pick top-2 largest axes as Y and X
    shape = arr.shape
    axes_sorted = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)
    y_ax, x_ax = axes_sorted[0], axes_sorted[1]
    # bring Y, X to front
    perm = [y_ax, x_ax] + [i for i in range(len(shape)) if i not in (y_ax, x_ax)]
    arr2 = np.transpose(arr, perm)
    if arr2.ndim == 2:
        return arr2[..., None]
    h, w = arr2.shape[0], arr2.shape[1]
    c = int(np.prod(arr2.shape[2:]))
    return arr2.reshape(h, w, c)


def convert_czi_to_tiff(czi_path, output_path, channel_index=None, ome=True, mppx=None, mppy=None, unit="µm"):
    """Convert single CZI file to (OME-)TIFF format.

    channel_index: optional int, choose specific channel (after flattening non-spatial dims)
    ome: write OME-TIFF metadata when True
    mppx/mppy: physical pixel size in microns; if None, try to parse from CZI metadata
    """
    try:
        with CziFile(czi_path) as czi:
            # Read image data -> YXC
            image_data = czi.asarray()
            image_yxc = _to_yxc(image_data)

            # Channel selection (only if not already standard 3-channel color)
            if channel_index is not None and image_yxc.shape[2] != 3:
                if image_yxc.shape[2] <= channel_index:
                    warnings.warn(f"Requested channel_index {channel_index} out of range; using 0")
                    channel_index = 0
                image_yxc = image_yxc[:, :, channel_index:channel_index+1]
            elif channel_index is not None and image_yxc.shape[2] == 3:
                warnings.warn("channel_index provided but input appears to be 3-channel color; keeping RGB intact.")

            # If more than 3 channels, keep first 3 for RGB visualization
            if image_yxc.shape[2] > 3:
                image_yxc = image_yxc[:, :, :3]

            # If single channel, DO NOT force replicate unless explicitly grayscale output desired
            # Keep as single-channel when not writing OME (classic TIFF) can be fine; but for OME RGB we need 3 samples.
            is_single_channel = (image_yxc.shape[2] == 1)

            # Normalize to 8-bit range for saving
            img = image_yxc.astype(np.float32)
            vmin, vmax = float(np.min(img)), float(np.max(img))
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin)
            img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            # Get physical pixel sizes
            if mppx is None or mppy is None:
                auto_mppx, auto_mppy = _parse_mpp_from_czi(czi)
                mppx = mppx if mppx is not None else auto_mppx
                mppy = mppy if mppy is not None else auto_mppy

            # Write TIFF (OME when requested)
            if ome:
                # tifffile expects RGB as samples-last with axes 'YXS'
                meta_phys = {}
                if mppx and mppy:
                    meta_phys = {
                        'PhysicalSizeX': float(mppx),
                        'PhysicalSizeXUnit': unit,
                        'PhysicalSizeY': float(mppy),
                        'PhysicalSizeYUnit': unit,
                    }
                try:
                    if is_single_channel:
                        meta_ome = {'axes': 'YX', **meta_phys}
                        tifffile.imwrite(str(output_path), img8[:, :, 0], photometric='minisblack', metadata=meta_ome, ome=True)
                    else:
                        meta_ome = {'axes': 'YXS', **meta_phys}
                        tifffile.imwrite(str(output_path), img8, photometric='rgb', metadata=meta_ome, ome=True)
                except Exception as e:
                    warnings.warn(f"OME-TIFF write failed ({e}); falling back to classic TIFF without OME metadata.")
                    tifffile.imwrite(str(output_path), img8[:, :, 0] if is_single_channel else img8,
                                      photometric=('minisblack' if is_single_channel else 'rgb'), metadata=None)
            else:
                # Classic TIFF without OME metadata; embed resolution if provided (in pixels per inch) as hint
                if mppx and mppy and mppx > 0 and mppy > 0:
                    # Convert microns per pixel to pixels per inch: ppi = 25400 / (um_per_px)
                    xres = 25400.0 / float(mppx)
                    yres = 25400.0 / float(mppy)
                    if is_single_channel:
                        tifffile.imwrite(str(output_path), img8[:, :, 0], photometric='minisblack', resolution=(xres, yres), metadata=None)
                    else:
                        tifffile.imwrite(str(output_path), img8, photometric='rgb', resolution=(xres, yres), metadata=None)
                else:
                    if is_single_channel:
                        tifffile.imwrite(str(output_path), img8[:, :, 0], photometric='minisblack', metadata=None)
                    else:
                        tifffile.imwrite(str(output_path), img8, photometric='rgb', metadata=None)
            return True
            
    except Exception as e:
        print(f"Error converting {czi_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert CZI files to (OME-)TIFF format with optional mpp metadata")
    parser.add_argument("--input", default="picture/", help="Input directory containing CZI files")
    parser.add_argument("--output", default="picture/tiff/", help="Output directory for TIFF files")
    parser.add_argument("--channel-index", type=int, default=None, help="Select specific channel index (default: keep first 3 or replicate 1 to RGB)")
    parser.add_argument("--ome", action="store_true", default=True, help="Write OME-TIFF with metadata (default: True)")
    parser.add_argument("--no-ome", dest="ome", action="store_false", help="Disable OME-TIFF; write classic TIFF")
    parser.add_argument("--mppx", type=float, default=None, help="Physical pixel size X (microns per pixel)")
    parser.add_argument("--mppy", type=float, default=None, help="Physical pixel size Y (microns per pixel)")
    parser.add_argument("--unit", type=str, default="µm", help="Physical size unit for OME-TIFF (default: µm)")
    
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
        
        if convert_czi_to_tiff(czi_file, output_file,
                               channel_index=args.channel_index,
                               ome=args.ome,
                               mppx=args.mppx,
                               mppy=args.mppy,
                               unit=args.unit):
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