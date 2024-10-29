
import os
import csv

# 设置 train 文件夹路径
train_folder_path = "/root/autodl-tmp/Proj/city_diffusion_demo/data2_2/train"  
csv_file = os.path.join(train_folder_path, "metadata.csv")  # metadata.csv in train folder

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["file_name", "text"])
    
    for folder_name in os.listdir(train_folder_path):
        folder_path = os.path.join(train_folder_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.endswith('_image'):
      
            city_name = folder_name.replace('_image', '')
            print(city_name)
     
            for file_name in os.listdir(folder_path):

                text = f"Buildings in {city_name}"
                
                full_file_name = os.path.join(folder_name,file_name)
                writer.writerow([full_file_name, text])
                print(file_name)

print(f"generate CSV  successfully：{csv_file}")
