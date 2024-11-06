import gradio as gr
from PIL import Image, ImageFilter
import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageFilter
from io import BytesIO
# class SaveImageEveryNStepsCallback:
#     def __init__(self, output_dir, total_steps, interval):
#         self.output_dir = output_dir
#         self.interval = interval
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         self.step_list = set(range(0, total_steps, interval))

#     def __call__(self, scheduler, **kwargs):
#         current_step = kwargs["step"]
#         if current_step in self.step_list:
#             image = kwargs["sample"].detach().cpu().squeeze().permute(1, 2, 0)
#             image = (image + 1) / 2  # normalize image
#             image = image.clamp(0, 1) * 255  # scale to 0-255
#             image = image.numpy().astype("uint8")
#             image_path = os.path.join(self.output_dir, f"image_at_step_{current_step}.png")
#             Image.fromarray(image).save(image_path)
#             print(f"Image saved at step {current_step}")
output_dir = "./saved_images"
# def save_image_callback(pipeline, i, t, latents, **kwargs):
#     interval = 5  # Save an image every 5 steps
#     if i % interval == 0:
#         # Convert latents to image
#         image = pipeline.decode_latents_to_image(latents)  # Adjust method call according to actual API
#         image = (image + 1) / 2 * 255
#         image = image.clip(0, 255).astype(np.uint8)
#         image = Image.fromarray(image)
#         # Save the image
#         image_path = os.path.join(output_dir, f"image_at_step_{i}.png")
#         image.save(image_path)
#         print(f"Image saved at step {i}")
        
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35),
    )
    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(image_array)

# Callback function to save images at specific intervals
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    image = latents_to_rgb(latents[0])
    image.save(f"./output_images/{step}.png")
    return callback_kwargs
# 加载预训练模型和权重
pipeline = AutoPipelineForText2Image.from_pretrained("bguisard/stable-diffusion-nano-2-1", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/root/autodl-tmp/Proj/city_diffusion_demo/root/autodl-tmp/sddata/finetune/lora/city_128/",weight_name="pytorch_lora_weights.safetensors")
def create_3d_surface_image(gray_image):
    # 将灰度图转换为 NumPy 数组，使用灰度值作为高度数据
    image_array = np.array(gray_image)
    x = np.arange(image_array.shape[1])
    y = np.arange(image_array.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = image_array  # 灰度值作为 Z 轴（高度）
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
    # 创建 3D 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='gray_r', edgecolor='none')
    facecolors = plt.cm.gray_r(Z_normalized)
    contour = ax.contourf(X, Y, Z, levels=100, cmap='gray_r')
    # ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False, edgecolor='none')

    # 设置标题和标签
    ax.set_title("3D Building Height Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height (Gray Value)")

    # 将图像保存到内存中的 PNG 文件
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)  # 关闭图形以释放内存

    # 从内存中加载图像并转换为 PIL 图像
    buf.seek(0)
    return Image.open(buf)

def process_image2(image):
  
    image = image.convert('L')
    image_array = np.array(image)
    image_array[image_array == 255] = 0
    
    # 反转像素值
    image_array = 255 - image_array
    image_array[image_array == 255] = 0
    image = Image.fromarray(image_array)
    return image

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from io import BytesIO

def create_3d_surface_image2(gray_image):
    # 将灰度图转换为 NumPy 数组，使用灰度值作为高度数据
    image_array = np.array(gray_image)
    x = np.arange(image_array.shape[1])
    y = np.arange(image_array.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = image_array

    # 创建 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将网格点展开为一维数组，以便 bar3d 绘制
    xpos, ypos = X.flatten(), Y.flatten()
    zpos = np.zeros_like(xpos)  # z轴从0开始

    # 设置每个像素的宽度和深度
    dx = dy = 1  # 每个像素的宽和深
    dz = Z.flatten()  # 高度值使用 Z 值

    # 使用灰度映射生成 RGBA 颜色
    normed_values = Z.flatten() / Z.max()  # 归一化到 0-1 之间
    colors = plt.cm.gray(normed_values)  # 使用灰度色图
    colors = colors[:, :3]  # 只保留 RGB 部分，不包括 alpha 通道
    colors = np.hstack([colors, np.ones((colors.shape[0], 1))])  # 添加 alpha 通道，确保是 RGBA 格式

    # 绘制实心长方体
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 将图像保存到内存中的 PNG 文件
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    # 从内存中加载图像并转换为 PIL 图像
    buf.seek(0)
    return Image.open(buf)


# 使用示例
# gray_image = <your grayscale image>
# final_image = create_solid_3d_surface_image(gray_image)
# final_image.show()

def generate_image(text,option):
    num_steps = 50
    interval = num_steps // 10
    output_dir = "./intermediate_images"
    # callback = SaveImageEveryNStepsCallback(output_dir, num_steps, interval)
    # generator = torch.manual_seed(42)
    # image = pipeline(text, num_inference_steps=num_steps, generator=generator, callback_on_step_end=decode_tensors,  
    # callback_on_step_end_tensor_inputs=["latents"])
    while True:
        image = pipeline(text, num_inference_steps=num_steps)
        final_image = image.images[0]
        final_pil_image = final_image.convert('L')
        final_image.save("output_images/f1.png")
        final_pil_image2 = final_pil_image.filter(ImageFilter.BLUR)
        final_pil_image2.save("output_images/ff2.png")
        # return final_pil_image2
        final_pil_image2 = process_image2(final_pil_image2)
              
        if option == "Ratio < 5":
            if calculate_building_ratio(final_image) < 5:
                # final_pil_image2 = final_pil_image.filter(ImageFilter.BLUR)
                # final_pil_image2.save("output_images/ff2.png")
                # # return final_pil_image2
                # final_pil_image2 = process_image2(final_pil_image2)
                return create_3d_surface_image2(final_pil_image2)
                # return convert_to_3d_image(final_pil_image)
        elif option == "Ratio >= 5":
            if calculate_building_ratio(final_image) >= 5:
               
                return create_3d_surface_image2(final_pil_image2)   
        else:
               
                return create_3d_surface_image2(final_pil_image2)     
        
        # final_pil_image = Image.fromarray((final_image.cpu().numpy() * 255).astype('uint8'))
          
    
   
    # Save as JPEG
    # image_path = os.path.join(output_dir, "final_image.jpg")
    # final_pil_image.save(image_path, "JPEG")

    return final_pil_image
    # return final_image
            
# def generate_image(text):
#     # 直接指定图片路径
#     image_path = '/root/autodl-tmp/Proj/city_diffusion_demo/images/beijing/beijing_0.png'
    
#     # 加载图片
#     image = Image.open(image_path)
    
#     return image

def check_requirements(image, requirement):
    # 根据选中的要求检查图片
    # 示例中的需求检查逻辑需要根据具体需求实现
    if requirement == "Option 1":
        # 检查条件1
        pass
    elif requirement == "Option 2":
        # 检查条件2
        pass
    return True  # 假设总是返回True

def generate_compliant_image(text, requirements):
    while True:
        image = generate_image(text)
        if check_requirements(image, requirements):
            break
    return image
def calculate_building_ratio(image):
    # 加载图片并转换为灰度图
    # img = Image.open(image_path).convert('L')
    img_array = np.array(image.convert('L'))
    print(img_array.size)
    # 建筑区域定义为所有非零像素
    building_area = np.count_nonzero(img_array != 255)

    # # 找到最高和最低的有建筑的像素行，用于估算楼层高度
    # non_zero_rows = np.nonzero(img_array)[0]
    # if non_zero_rows.size == 0:
    #     return 0  # 如果没有建筑，则返回0
    # min_row, max_row = np.min(non_zero_rows), np.max(non_zero_rows)
    # height = max_row - min_row + 1
    height = np.sum(img_array[img_array != 255])
    print(height)
    print(img_array[img_array != 255])
    # 估算楼层数，假设每层楼高3米
    floors = height / 3
    # 计算非255的像素数量
    # 计算同时不是255且不是0的像素数量
    building_area = np.count_nonzero((img_array != 255) & (img_array != 0))

    (print(building_area))

    total_area = img_array.size
    print(total_area)
    # 计算比例：建筑的底面积 * 楼层 / 地块的总面积
    ratio = (floors) / total_area
    print(ratio)
    print(floors)
    return ratio/9



iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Dropdown(choices=["Ratio < 5", "Ratio >= 5", "No ratio restriction"], label="Select Ratio Requirement")
    ],
    outputs="image",
    title="Image of Buildings Generation",
    description="Enter text and specify requirements for the generated image. The image will be regenerated until it meets the requirements."
)

iface.launch(share=True)