import pyvips

def big_tiff_to_png(input_path, output_path):
    image = pyvips.Image.new_from_file(input_path, access="sequential")
    image.write_to_file(output_path)

if __name__ == "__main__":
    big_tiff_to_png("/home/sec312/tsgh/unet_mask/tile/tile_x60928_y30720_dish.tiff", "/home/sec312/tsgh/unet_mask/tile/tile_x60928_y30720_dish.png")
