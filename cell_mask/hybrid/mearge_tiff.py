import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tifffile import imread, imwrite
import cv2

def process_file(args):
    """處理單一檔案"""
    dish_path, her2_path, output_path = args
    dish_img = imread(dish_path)
    her2_img = imread(her2_path)
    if dish_img.shape != her2_img.shape:
        her2_img = cv2.resize(her2_img, (dish_img.shape[1], dish_img.shape[0]))
    merged = cv2.addWeighted(dish_img, 0.5, her2_img, 0.5, 0)
    imwrite(output_path, merged)
    return True

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dish_dir = os.path.join(script_dir, "dish")
    her2_dir = os.path.join(script_dir, "her2")
    output_dir = os.path.join(script_dir, "merge")
    os.makedirs(output_dir, exist_ok=True)

    # 準備任務列表
    tasks = []
    for f in os.listdir(dish_dir):
        if f.endswith(('.tiff', '.tif')):
            her2_path = os.path.join(her2_dir, f)
            if os.path.exists(her2_path):
                tasks.append((
                    os.path.join(dish_dir, f),
                    her2_path,
                    os.path.join(output_dir, f)
                ))

    print(f"處理 {len(tasks)} 個檔案...")
    
    # 使用多進程加速
    with ProcessPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(process_file, t) for t in tasks]
        done = 0
        for _ in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(tasks)}")
    
    print("完成!")

if __name__ == "__main__":
    main()