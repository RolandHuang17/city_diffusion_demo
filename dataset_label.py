
import os

root_dir = 'buildings'
image_dirs = ['Beijing_image', 'Guangzhou_image', 'Hongkong_image', 'Wuhan_image']

for target_dir in image_dirs:
    img_path = os.listdir(os.path.join(root_dir, target_dir))
    label = target_dir.split('_')[0]  # 获取文件夹名称的第一部分作为标签
    out_dir = target_dir.replace('image', 'label')  # 将 image 替换为 label 作为输出文件夹名称

    # 确保输出的标签文件夹存在
    os.makedirs(os.path.join(root_dir, out_dir), exist_ok=True)

    for i in img_path:
        # 对每张图片生成对应的标签文件
        file_name = i.split('.TIF')[0]  # 假设文件都是 .TIF 格式
        with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
            f.write(label)  # 将标签写入对应的 txt 文件
