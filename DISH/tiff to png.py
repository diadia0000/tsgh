import pyvips


def big_tiff_to_png(input_path, output_path):
    # 讀取大圖
    image = pyvips.Image.new_from_file(input_path, access="sequential")
    # 轉換成灰階 (black-white)
    gray = image.colourspace("b-w")
    # 輸出 PNG
    gray.write_to_file(output_path)

if __name__ == "__main__":
    big_tiff_to_png("DISH_final_stitched_white_bg_final.tiff", "DISH_gray.png")
