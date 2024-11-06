import gradio as gr
from PIL import Image
import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
import os
from PIL import ImageOps


# 或者使用 image.save("path/to/save/image.png")
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
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)  # Change the order of dimensions

    return Image.fromarray(image_array)

# Callback function to save images at specific intervals
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    # image = latents_to_rgb(latents)
    image = pipe.vae.decode(latents/pipe.vae.config.scaling_factor)[0]
    image = pipe.image_processor.postprocess(image,output_type = "pil")[0]
    os.makedirs("./output_images",exist_ok=True)
    image.save(f"./output_images/{step}.png")
    return callback_kwargs
# 加载预训练模型和权重
pipeline = AutoPipelineForText2Image.from_pretrained("bguisard/stable-diffusion-nano-2-1", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/root/autodl-tmp/Proj/city_diffusion_demo/root/autodl-tmp/sddata/finetune/lora/city_128/",weight_name="pytorch_lora_weights.safetensors")
def process_image2(image):
  
    image = image.convert('L')
    image_array = np.array(image)
    image_array[image_array == 255] = 0
    
    # 反转像素值
    image_array = 255 - image_array
    image = Image.fromarray(image_array)
    return image

def process_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image_array = np.array(image)
    image_array[image_array == 255] = 0
    
    # 反转像素值
    image_array = 255 - image_array
    
    # 将NumPy数组转换回图像
    processed_image = Image.fromarray(image_array)
    processed_path = image_path.replace(".png", "_processed.png")
    processed_image.save(processed_path)
    return processed_path
def generate_image(text, option):
    num_steps = 50
    interval = num_steps // 10
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    final_image = None
    image = pipeline(text, num_inference_steps=num_steps, callback_on_step_end=decode_tensors, callback_on_step_end_tensor_inputs=["latents"])
    final_image = image.images[0]
    # 生成图像的逻辑
    for step in range(num_steps):
        
        # if step == num_steps - 1:
        #     final_image = process_image2(final_image)
        if step % interval == 0:
            image_path = os.path.join(output_dir, f"{step}.png")
            image_paths.append(process_image(image_path))

    # 检查图像是否满足建筑比例要求
    if (option == "Ratio < 5" and calculate_building_ratio(final_image) < 5) or \
       (option == "Ratio >= 5" and calculate_building_ratio(final_image) >= 5):
        final_image = process_image2(final_image)   
        return final_image, image_paths
    else:
        return generate_image(text, option)  # 重新生成直到满足条件
# def generate_image(text,option):
#     num_steps = 50
#     interval = num_steps // 10
#     output_dir = "./output_images"
#     # callback = SaveImageEveryNStepsCallback(output_dir, num_steps, interval)
#     # generator = torch.manual_seed(42)
#     # image = pipeline(text, num_inference_steps=num_steps, generator=generator, callback_on_step_end=decode_tensors,  
#     # callback_on_step_end_tensor_inputs=["latents"])
#     while True:
#         image = pipeline(text, num_inference_steps=num_steps, callback_on_step_end=decode_tensors,callback_on_step_end_tensor_inputs=["latents"])
#         final_image = image.images[0]
#         final_image = ImageOps.invert(final_image)
#         image_paths = []
#         if option == "Ratio < 5":
#             if calculate_building_ratio(final_image) < 5:
#                 final_pil_image = final_image.convert('L')
#                 for step in range(num_steps):
#                     if step % interval == 0:
            
#                         image_path = os.path.join(output_dir, f"{step}.png")
#                         image_paths.append(image_path)  # Store image path
#                         processed_paths = [process_image(path) for path in image_paths]
#                 # return final_pil_image           
#             # return image_paths
#             return processed_paths
#         else:
#             if calculate_building_ratio(final_image) >= 5:
#                 final_pil_image = final_image.convert('L')
#                 for step in range(num_steps):
#                     if step % interval == 0:
            
#                         image_path = os.path.join(output_dir, f"{step}.png")
#                         image_paths.append(image_path)  # Store image path
#                         processed_paths = [process_image(path) for path in image_paths]
#                 # return final_pil_image           
#             # return image_paths
#             return processed_paths
#         # final_pil_image = Image.fromarray((final_image.cpu().numpy() * 255).astype('uint8'))
          
    
   
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
    return ratio/10



iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Dropdown(choices=["Ratio < 5", "Ratio >= 5"], label="Select Ratio Requirement")
    ],
    # outputs="image",
    outputs=[gr.Image(label="Final Image"), gr.Gallery(label="Processed Intermediate Images")],

    # outputs=gr.Gallery(label="Intermediate Images"),
    title="Image of Buildings Generation",
    description="Enter text and specify requirements for the generated image. The image will be regenerated until it meets the requirements."
)

iface.launch(share=True)
