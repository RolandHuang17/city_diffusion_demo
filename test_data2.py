import cv2
import numpy as np
import os
from PIL import Image
# Constants for the different read modes
IMREAD_UNCHANGED = -1  # Keep the image in its original format, including bit depth and number of channels.
IMREAD_GRAYSCALE = 0  # Convert image to grayscale. If the source image is 16-bit, the output will be 8-bit.
IMREAD_COLOR = 1  # Convert image to RGB format with 8-bit depth per channel.
IMREAD_ANYDEPTH = 2  # Preserve the bit depth of the image but convert it to grayscale.
IMREAD_ANYCOLOR = 4  # If the image has 3 or fewer channels, keep them; if more, use only the first three. Convert to 8-bit depth.

# Example usage
# filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data/train/Wuhan_image/WH325.TIF"
# output_filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data2/train/Wuhan_image/WH325.jpg"

# filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data/train/Guangzhou_image/GZ88.TIF"
# output_filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data2/train/Guangzhou_image/GZ88.jpg" 
filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data2_2/data2_2/train/Guangzhou_image/GZ19226.png"
# output_filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data2/train/Beijing_image/BJ91.jpg" 
# filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data/train/Hongkong_image/HK5115.TIF"
# output_filename = "/root/autodl-tmp/Proj/city_diffusion_demo/data2/train/Hongkong_image/HK5115.jpg"
# os.makedirs(os.path.dirname(output_filename),exist_ok=True)
# Read the image with no changes
IMREAD_UNCHANGED = -1
# image_unchanged = cv2.imread(filename,IMREAD_ANYDEPTH)
image_unchanged = Image.open(filename).convert('L')
print(image_unchanged)
image_unchanged = np.array(image_unchanged)
# original_depth = image_unchanged.dtype
# print(f"Original depth: {original_depth}")
# image_unchanged = cv2.convertScaleAbs(image_unchanged)
# image_unchanged = cv2.normalize(image_unchanged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(255-image_unchanged)
# If the image is grayscale, convert to RGB
# if len(image_unchanged.shape) == 2:  # If the image is grayscale
#         image_rgb = cv2.cvtColor(image_unchanged, cv2.COLOR_GRAY2RGB)
# else:
#         image_rgb = image_unchanged
# if len(image_unchanged.shape) == 2:  # If the image is grayscale
#         image_rgb = cv2.cvtColor(image_unchanged, cv2.COLOR_GRAY2RGB)
# normalized_image = cv2.normalize(image_unchanged, None, 0, 255, cv2.NORM_MINMAX)
# image_rgb = image_rgb.astype(original_depth)
# print(f"Restored depth: {image_rgb.dtype}")
# cv2.imwrite(output_filename, image_rgb)
# image_rgb = image_rgb.convert('L')
np_array = np.array(image_unchanged,dtype=np.uint8)
# print("Numpy Array Shape:", np_array.shape)

# Find unique values and their counts
unique_values, counts = np.unique(np_array, return_counts=True)

# 输出每个独特值及其出现的次数
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

# Print the number of unique values
print(f"Number of unique values: {unique_values}")
print(np_array.shape)