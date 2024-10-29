import cv2
import os
import numpy as np

train_folder_path = "/root/autodl-tmp/Proj/city_diffusion_demo/data/train" 
output_folder_path = "/root/autodl-tmp/Proj/city_diffusion_demo/data3/train" 
for folder_name in os.listdir(train_folder_path):
    folder_path = os.path.join(train_folder_path, folder_name)
    output_file_folder_path = os.path.join(output_folder_path,folder_name)
    os.makedirs(output_file_folder_path,exist_ok=True)
    if os.path.isdir(folder_path):
       
        for file_name in os.listdir(folder_path):
            print(file_name)
            if file_name.endswith('.TIF'):
                # 构建完整的输入文件路径
                print("Yes")
                full_file_name = os.path.join(folder_path, file_name)
                print(full_file_name)
                # 读取 .tif 图像
                # image = cv2.imread(full_file_name, cv2.IMREAD_UNCHANGED)
                IMREAD_UNCHANGED = -1
                image_unchanged = cv2.imread(full_file_name, IMREAD_UNCHANGED)
                image_unchanged = cv2.convertScaleAbs(image_unchanged)
                if len(image_unchanged.shape) == 2:  # If the image is grayscale
                    image_rgb = cv2.cvtColor(image_unchanged, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image_unchanged
                # if len(image_unchanged.shape) == 2:  # If the image is grayscale
                #         image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
                # 检查图像是否成功读取
                # if image is None:
                #     print(f"Failed to read the image from {full_file_name}")
                #     continue
                print(file_name)
                # 构建输出文件路径
                output_file_name = os.path.splitext(file_name)[0] + '.jpg'
                output_file_path = os.path.join(output_file_folder_path, output_file_name)
                
                # 保存图像为 .jpg 格式
                success = cv2.imwrite(output_file_path, image_rgb)    
                