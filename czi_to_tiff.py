import os
import numpy as np
import pyvips
from aicspylibczi import CziFile
from pathlib import Path
import math
from tqdm import tqdm
import gc
import tifffile

class CziToTiffConverter:
    """Convert large CZI files to pyramidal TIFF with tile-based processing"""

    def __init__(self, input_dir, output_dir, tile_count=48):
        """
        Initialize converter

        Args:
            input_dir: Directory containing CZI files
            output_dir: Directory to save TIFF files
            tile_count: Number of tiles to split the image into (default: 48)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tile_count = tile_count

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp directory for intermediate files
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Optimize pyvips settings for large images
        pyvips.cache_set_max(1000)  # Increase cache size (default is 100)
        pyvips.cache_set_max_mem(2 * 1024 * 1024 * 1024)  # 2GB memory cache
        pyvips.cache_set_max_files(200)  # Allow more open files

    def get_czi_dimensions(self, czi_path):
        """Get dimensions of CZI file"""
        czi = CziFile(czi_path)

        # Get bounding box
        bbox = czi.get_mosaic_bounding_box()
        width = bbox.w
        height = bbox.h
        x_start = bbox.x
        y_start = bbox.y

        print(f"Image dimensions: {width} x {height}")
        print(f"Image bounding box: x={x_start}, y={y_start}, w={width}, h={height}")
        return width, height, x_start, y_start, czi

    def calculate_tile_regions(self, width, height, x_start, y_start, tile_count):
        """
        Calculate tile regions for processing

        Args:
            width: Image width
            height: Image height
            x_start: Starting X coordinate in CZI
            y_start: Starting Y coordinate in CZI
            tile_count: Number of tiles to create

        Returns:
            List of (x, y, w, h) tuples for each tile region
        """
        # Calculate grid dimensions (e.g., 6x8 = 48 tiles)
        aspect_ratio = width / height
        rows = int(math.sqrt(tile_count / aspect_ratio))
        cols = int(tile_count / rows)

        # Adjust to ensure we have at least tile_count tiles
        while rows * cols < tile_count:
            cols += 1

        print(f"Grid layout: {rows} rows x {cols} columns = {rows * cols} tiles")

        tile_width = width // cols
        tile_height = height // rows

        regions = []
        for row in range(rows):
            for col in range(cols):
                # Calculate relative position
                x_offset = col * tile_width
                y_offset = row * tile_height
                w = tile_width if col < cols - 1 else width - x_offset
                h = tile_height if row < rows - 1 else height - y_offset

                # Add CZI's starting coordinates
                x = x_start + x_offset
                y = y_start + y_offset

                regions.append((x, y, w, h))

        return regions, rows, cols, tile_width, tile_height

    def bgr_to_rgb(self, image_array):
        """Convert BGR image to RGB"""
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Swap channels: BGR -> RGB
            return image_array[:, :, ::-1].copy()
        return image_array

    def process_tile(self, czi, region):
        """
        Process a single tile from CZI file

        Args:
            czi: CziFile object
            region: (x, y, w, h) tuple defining the region

        Returns:
            numpy array of the tile image
        """
        x, y, w, h = region

        # Read the region from CZI file
        # CziFile uses (start_x, start_y, width, height) format
        tile_data = czi.read_mosaic(region=(x, y, w, h), scale_factor=1.0, C=0)

        # Remove extra dimensions (like C, Z, T if they exist)
        # Expected shape might be like (1, 1, H, W, C) or (H, W, C)
        while len(tile_data.shape) > 3:
            tile_data = np.squeeze(tile_data, axis=0)

        # Convert BGR to RGB
        tile_data = self.bgr_to_rgb(tile_data)

        return tile_data

    def create_pyramidal_tiff(self, czi_path, output_path):
        """
        Convert CZI to pyramidal TIFF using direct tile writing approach

        Args:
            czi_path: Path to input CZI file
            output_path: Path to output TIFF file
        """
        print(f"\nProcessing: {czi_path.name}")
        print("="*60)

        # Get CZI dimensions
        width, height, x_start, y_start, czi = self.get_czi_dimensions(czi_path)

        # Calculate tile regions
        regions, rows, cols, tile_width, tile_height = self.calculate_tile_regions(
            width, height, x_start, y_start, self.tile_count
        )

        print(f"Tile size: {tile_width} x {tile_height}")
        print(f"Processing {len(regions)} tiles...")

        # Determine number of channels from first tile
        print("\nDetecting image properties...")
        first_region = regions[0]
        first_tile = self.process_tile(czi, first_region)
        bands = first_tile.shape[2] if len(first_tile.shape) == 3 else 1
        dtype = first_tile.dtype

        print(f"Image properties: {width} x {height} x {bands} bands, dtype={dtype}")

        # Create temporary full-resolution TIFF with tiled storage
        temp_full_res = self.temp_dir / f"{czi_path.stem}_full_res.tiff"

        print("\n[Stage 1/2] Writing tiles directly to TIFF using memory mapping...")

        # Create a memory-mapped TIFF file that we can write to directly
        # This allows us to write tiles without loading the entire image into memory
        print("Creating memory-mapped TIFF file...")

        # Create a memory-mapped array
        memmap_file = self.temp_dir / f"{czi_path.stem}_memmap.dat"
        full_array = np.memmap(
            memmap_file,
            dtype=dtype,
            mode='w+',
            shape=(height, width, bands)
        )

        print(f"Memory-mapped array created: {height} x {width} x {bands}")

        # Write tiles directly into the memory-mapped array
        for tile_idx in tqdm(range(len(regions)), desc="Writing tiles"):
            row = tile_idx // cols
            col = tile_idx % cols

            if tile_idx < len(regions):
                region = regions[tile_idx]
                tile_array = self.process_tile(czi, region)

                # Calculate position in the full image
                y_start_pos = row * tile_height
                x_start_pos = col * tile_width

                # Get actual tile dimensions
                tile_h, tile_w = tile_array.shape[:2]

                # Calculate end positions (handle edge cases)
                y_end_pos = min(y_start_pos + tile_h, height)
                x_end_pos = min(x_start_pos + tile_w, width)

                # Crop tile if needed to fit exact boundaries
                tile_h_actual = y_end_pos - y_start_pos
                tile_w_actual = x_end_pos - x_start_pos

                # Write tile directly to memory-mapped array
                full_array[y_start_pos:y_end_pos, x_start_pos:x_end_pos, :] = \
                    tile_array[:tile_h_actual, :tile_w_actual, :]

                # Free memory
                del tile_array

                # Flush to disk periodically (every 10 tiles)
                if tile_idx % 10 == 0:
                    full_array.flush()

        # Final flush
        full_array.flush()

        print(f"Writing memory-mapped data to TIFF file...")

        # Write the complete image as a tiled TIFF
        tifffile.imwrite(
            temp_full_res,
            full_array,
            bigtiff=True,
            tile=(256, 256),
            compression='lzw',
            photometric='rgb' if bands == 3 else 'minisblack'
        )

        # Clean up memory-mapped file
        del full_array
        gc.collect()

        if memmap_file.exists():
            memmap_file.unlink()

        print(f"\n✓ Saved full resolution TIFF: {temp_full_res.name}")

        # Stage 2: Convert to pyramidal TIFF using pyvips
        print("\n[Stage 2/2] Creating pyramidal TIFF...")
        print("Loading full resolution image...")

        full_image = pyvips.Image.new_from_file(str(temp_full_res), access='sequential')

        print(f"Complete image size: {full_image.width} x {full_image.height}")
        print("Generating pyramid levels...")
        print("Zoom levels: 1.0, 0.5, 0.25, 0.125, 0.0625")

        full_image.tiffsave(
            str(output_path),
            compression='jpeg',
            Q=90,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            bigtiff=True,
            properties=True,
            xres=2.0,
            yres=2.0,
        )

        # Clean up temporary file
        print("\nCleaning up temporary files...")
        if temp_full_res.exists():
            temp_full_res.unlink()

        print(f"✓ Successfully saved: {output_path.name}")
        print(f"  Output size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

    def convert_all(self):
        """Convert all CZI files in input directory"""
        # Find all CZI files
        czi_files = list(self.input_dir.glob("*.czi"))

        if not czi_files:
            print(f"No CZI files found in {self.input_dir}")
            return

        print(f"Found {len(czi_files)} CZI files to convert")
        print("="*60)

        for czi_path in czi_files:
            # Create output filename (replace .czi with .tiff)
            output_name = czi_path.stem + ".tiff"
            output_path = self.output_dir / output_name

            # Delete old file if exists (following user's coding instructions)
            if output_path.exists():
                print(f"Removing existing file: {output_path.name}")
                output_path.unlink()

            try:
                self.create_pyramidal_tiff(czi_path, output_path)
            except Exception as e:
                print(f"✗ Error processing {czi_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Clean up temp directory
        try:
            if self.temp_dir.exists():
                self.temp_dir.rmdir()
        except:
            pass

        print("\n" + "="*60)
        print("Conversion complete!")


def main():
    """Main function"""
    # Define paths
    input_dir = r"E:\Class\tsgh\picture\whole_size"
    output_dir = r"E:\Class\tsgh\picture\WSI"

    # Create converter with 480 tiles (increased from 48 for better performance)
    converter = CziToTiffConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_count=48
    )

    # Convert all CZI files
    converter.convert_all()


if __name__ == "__main__":
    main()
