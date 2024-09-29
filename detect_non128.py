from PIL import Image
import os

def check_image_sizes(root_dir, label_dirs, target_size=(128, 128)):
    incorrect_images = []

    for label_dir in label_dirs:
        dir_path = os.path.join(root_dir, label_dir)
        img_files = os.listdir(dir_path)

        for img_file in img_files:
            img_path = os.path.join(dir_path, img_file)
            try:
                with Image.open(img_path) as img:
                    # 检查图像大小
                    if img.size != target_size:
                        print(f"图像 {img_file} 的大小为 {img.size}，不是目标大小 {target_size}")
                        incorrect_images.append(img_path)
            except Exception as e:
                print(f"无法打开图像 {img_file}: {e}")

    return incorrect_images

# 调用函数，检查哪些图像的大小不是 128x128
root_dir = "buildings"
label_dirs = ["Beijing_image", "Guangzhou_image", "Hongkong_image", "Wuhan_image"]
incorrect_images = check_image_sizes(root_dir, label_dirs, target_size=(128, 128))

# 输出不符合要求的图像
if incorrect_images:
    print(f"以下图像大小不符合 {128}x{128} 尺寸要求:")
    for img_path in incorrect_images:
        print(img_path)
else:
    print("所有图像的尺寸都是 128x128")