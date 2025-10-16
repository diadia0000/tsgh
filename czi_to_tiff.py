import os
import numpy as np
import pyvips
from aicspylibczi import CziFile
from pathlib import Path
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def _process_tile_worker(args):
    """Worker function for multiprocessing"""
    czi_path, x, y, w, h = args
    czi = CziFile(czi_path)
    tile_data = czi.read_mosaic(region=(x, y, w, h), C=0, scale_factor=1)

    while len(tile_data.shape) > 3:
        tile_data = np.squeeze(tile_data, axis=0)

    if len(tile_data.shape) == 3 and tile_data.shape[2] == 3:
        tile_data = tile_data[:, :, ::-1]

    return tile_data

class CziToTiffConverter:
    """Convert CZI files to compressed pyramidal TIFF"""

    def __init__(self, input_dir: str, output_dir: str, tile_size: int = 4096):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Configure pyvips for memory efficiency
        pyvips.cache_set_max(0)  # Disable operation cache
        pyvips.cache_set_max_mem(100 * 1024 * 1024)  # 100MB memory limit
        pyvips.cache_set_max_files(0)  # Disable file cache

    def create_pyramidal_tiff(self, czi_path: Path, output_path: Path) -> None:
        """Convert CZI to compressed pyramidal TIFF with tile-based processing"""
        print(f"\nProcessing: {czi_path.name}")
        print("="*60)

        czi = CziFile(czi_path)
        bbox = czi.get_mosaic_bounding_box()
        width, height = bbox.w, bbox.h
        x_start, y_start = bbox.x, bbox.y

        print(f"Dimensions: {width} x {height}")

        cols = math.ceil(width / self.tile_size)
        rows = math.ceil(height / self.tile_size)
        print(f"Processing {rows}x{cols} tiles ({rows*cols} total)")

        print("Processing tiles...")

        # Save tiles as temp vips files
        temp_tiles = []
        for row in tqdm(range(rows), desc="Rows"):
            tasks = [(czi_path, x_start + col * self.tile_size, y_start + row * self.tile_size,
                     min(self.tile_size, width - col * self.tile_size),
                     min(self.tile_size, height - row * self.tile_size))
                     for col in range(cols)]

            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                tile_arrays = list(executor.map(_process_tile_worker, tasks))

            # Save row tiles to disk
            row_tile_files = []
            for col, tile_data in enumerate(tile_arrays):
                tile_h, tile_w = tile_data.shape[:2]
                bands = tile_data.shape[2] if len(tile_data.shape) == 3 else 1

                tile_file = self.temp_dir / f"tile_r{row}_c{col}.v"
                vips_tile = pyvips.Image.new_from_memory(
                    tile_data.tobytes(), tile_w, tile_h, bands, 'uchar'
                )
                vips_tile.write_to_file(str(tile_file))
                row_tile_files.append(tile_file)

            temp_tiles.append(row_tile_files)

        print("Assembling image...")
        # Join tiles from disk
        row_images = []
        for row_files in tqdm(temp_tiles, desc="Joining rows"):
            tiles = [pyvips.Image.new_from_file(str(f), access='sequential') for f in row_files]
            row_img = pyvips.Image.arrayjoin(tiles, across=len(tiles))

            # Save row to disk
            row_file = self.temp_dir / f"row_{len(row_images)}.v"
            row_img.write_to_file(str(row_file))
            row_images.append(row_file)

            # Release objects
            del tiles, row_img

            # Cleanup tile files
            for f in row_files:
                try:
                    f.unlink()
                except:
                    pass

        # Join rows from disk
        print("Joining final image...")
        rows = [pyvips.Image.new_from_file(str(f), access='sequential') for f in row_images]
        full_img = pyvips.Image.arrayjoin(rows, across=1)

        print("Saving with pyramid...")
        full_img.tiffsave(
            str(output_path),
            compression='jpeg',
            Q=70,
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            bigtiff=True
        )

        # Release objects
        del rows, full_img

        # Cleanup
        for f in row_images:
            try:
                if f.exists():
                    f.unlink()
            except:
                pass

        size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"✓ Saved: {output_path.name} ({size_gb:.2f} GB)")

    def convert_all(self) -> None:
        """Convert all CZI files"""
        czi_files = list(self.input_dir.glob("*.czi"))

        if not czi_files:
            print(f"No CZI files found in {self.input_dir}")
            return

        print(f"Found {len(czi_files)} CZI files")

        for czi_path in czi_files:
            output_path = self.output_dir / f"{czi_path.stem}.tiff"

            if output_path.exists():
                print(f"Removing existing: {output_path.name}")
                output_path.unlink()

            try:
                self.create_pyramidal_tiff(czi_path, output_path)
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()

        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    converter = CziToTiffConverter(
        input_dir=r"E:\Class\tsgh\picture\whole_size\40X",
        output_dir=r"E:\Class\tsgh\picture\WSI\40X"
    )
    converter.convert_all()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
