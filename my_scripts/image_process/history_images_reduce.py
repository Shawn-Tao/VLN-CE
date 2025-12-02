# 将 指定路径下，每个文件夹中的图像分辨率由424x240 缩放到 112x56 使用cubic插值，并保存在指定路径的的同名文件夹下
import os
from PIL import Image
import argparse
import glob
import cv2
import tqdm 

# 获取指定目录下的所有文件夹
def get_subdirectories(directory):
    """
    Get all subdirectories in the specified directory.
    
    Args:
        directory (str): Path to the directory to search for subdirectories.
    
    Returns:
        list: List of subdirectory paths.
    """
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# 获取指定目录下的所有 jpg 图像文件
def get_image_files(directory):
    """
    Get all jpg image files in the specified directory.
    
    Args:
        directory (str): Path to the directory to search for jpg files.
    
    Returns:
        list: List of jpg file paths.
    """
    return glob.glob(os.path.join(directory, '*.jpg'))
    
def main():
    parser = argparse.ArgumentParser(description='Resize images in subdirectories.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing subdirectories with images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save resized images.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # 获取所有子目录
    subdirs = get_subdirectories(input_dir)
    
    print(f"Found subdirectories: {subdirs}")

    # for subdir in subdirs:
    #     input_subdir_path = os.path.join(input_dir, subdir)
    #     output_subdir_path = os.path.join(output_dir, subdir)

    #     # 创建输出子目录
    #     os.makedirs(output_subdir_path, exist_ok=True)

    #     # 获取所有 jpg 图像文件
    #     image_files = get_image_files(input_subdir_path)

    #     for image_file in image_files:
    #         img = Image.open(image_file)
    #         img_resized = img.resize((112, 56), Image.Resampling.BICUBIC)
    #         output_image_path = os.path.join(output_subdir_path, os.path.basename(image_file))
    #         img_resized.save(output_image_path)
    
    # unsing tqdm to show progress bar
    for subdir in tqdm.tqdm(subdirs, desc="Processing subdirectories"):
        input_subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, subdir)

        # 创建输出子目录
        os.makedirs(output_subdir_path, exist_ok=True)

        # 获取所有 jpg 图像文件
        image_files = get_image_files(input_subdir_path)

        for image_file in tqdm.tqdm(image_files, desc=f"Processing images in {subdir}"):
            img = Image.open(image_file)
            # img_resized = img.resize((112, 56), Image.Resampling.BICUBIC)
            img_resized = img.resize((112, 56), resample=Image.LANCZOS)
            output_image_path = os.path.join(output_subdir_path, os.path.basename(image_file))
            img_resized.save(output_image_path)


if __name__ == "__main__":
    main()