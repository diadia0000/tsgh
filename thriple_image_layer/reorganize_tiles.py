#!/usr/bin/env python3
"""
重新组织 tiles 到子资料夹
每个子资料夹包含 10000 张图片
"""

import os
import shutil
from pathlib import Path


def reorganize_tiles(source_dir, tiles_per_folder=10000):
    """
    将图片文件重新组织到子文件夹中
    
    Args:
        source_dir: 源目录路径
        tiles_per_folder: 每个子文件夹的图片数量
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"目录不存在: {source_dir}")
        return
    
    # 获取所有图片文件
    image_files = sorted([f for f in source_path.iterdir() if f.is_file()])
    total_files = len(image_files)
    
    print(f"\n处理目录: {source_dir}")
    print(f"总文件数: {total_files}")
    print(f"每个子文件夹: {tiles_per_folder} 张")
    print(f"将创建: {(total_files + tiles_per_folder - 1) // tiles_per_folder} 个子文件夹")
    
    # 创建临时目录来存放重新组织的文件
    temp_dir = source_path.parent / f"{source_path.name}_temp"
    temp_dir.mkdir(exist_ok=True)
    
    # 移动文件到子文件夹
    for idx, file_path in enumerate(image_files):
        # 计算子文件夹编号（从 1 开始）
        folder_num = idx // tiles_per_folder + 1
        subfolder_name = f"batch_{folder_num:03d}"
        
        # 创建子文件夹
        subfolder_path = temp_dir / subfolder_name
        subfolder_path.mkdir(exist_ok=True)
        
        # 移动文件
        dest_path = subfolder_path / file_path.name
        shutil.move(str(file_path), str(dest_path))
        
        # 显示进度
        if (idx + 1) % 1000 == 0:
            print(f"  进度: {idx + 1}/{total_files} ({(idx + 1) / total_files * 100:.1f}%)")
    
    # 删除原始空目录，并重命名临时目录
    source_path.rmdir()
    temp_dir.rename(source_path)
    
    print(f"✓ 完成: {source_dir}")


def main():
    """主函数"""
    base_dir = "/home/sec312/tsgh/thriple_image_layer/output/tiles_lv1"
    
    # 要处理的三个目录
    directories = [
        os.path.join(base_dir, "her2"),
        os.path.join(base_dir, "dish"),
        os.path.join(base_dir, "merged")
    ]
    
    print("=" * 60)
    print("重新组织 Tiles 到子资料夹")
    print("=" * 60)
    
    for directory in directories:
        reorganize_tiles(directory, tiles_per_folder=10000)
    
    print("\n" + "=" * 60)
    print("所有目录处理完成！")
    print("=" * 60)
    
    # 显示最终结构
    print("\n最终目录结构:")
    for directory in directories:
        print(f"\n{directory}:")
        dir_path = Path(directory)
        if dir_path.exists():
            subfolders = sorted([d for d in dir_path.iterdir() if d.is_dir()])
            for subfolder in subfolders:
                file_count = len(list(subfolder.iterdir()))
                print(f"  {subfolder.name}: {file_count} 张图片")


if __name__ == "__main__":
    main()
