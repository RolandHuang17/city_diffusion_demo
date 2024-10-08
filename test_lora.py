from diffusers import AutoPipelineForText2Image
import torch
import os
pipeline = AutoPipelineForText2Image.from_pretrained("bguisard/stable-diffusion-nano-2-1", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("./root/autodl-tmp/sddata/finetune/lora/city_128/checkpoint-15000/", weight_name="pytorch_lora_weights.safetensors")
for i in range(10):
    image = pipeline("Buildings in Beijing").images[0]
    image.save("./beijing.png")
    
# 定义城市列表
cities = ["Beijing", "Guangzhou", "Hongkong", "Wuhan"]

# 确保保存图片的目录存在
output_dir = "./images"
os.makedirs(output_dir, exist_ok=True)

# 生成并保存图片
for city in cities:
    # 为每个城市创建一个单独的目录
    city_dir = os.path.join("./images", city.lower())
    os.makedirs(city_dir, exist_ok=True)
    
    for i in range(10):
        image = pipeline(f"Buildings in {city}").images[0]
        image.save(os.path.join(city_dir, f"{city.lower()}_{i}.png"))