import pyvips

def big_tiff_to_png(input_path, output_path):
    image = pyvips.Image.new_from_file(input_path, access="sequential")
    image.write_to_file(output_path)

if __name__ == "__main__":
    big_tiff_to_png("picture/whole_size/two_image.tiff", "two_image.png")
