import os
from datasets import load_dataset

data_dir="/root/autodl-tmp/Proj/city_diffusion_demo/data"
data_files = {}
data_files["train"] = os.path.join(data_dir, "**")
dataset = load_dataset(
    "imagefolder",
    data_files=data_files,
    
        )

# dataset = load_dataset("imagefolder", data_dir="/root/autodl-tmp/Proj/city_diffusion_demo/data",drop_labels=False)
print(dataset["train"].column_names)

print(dataset["train"][-1])
# from diffusers import StableDiffusionPipeline

# pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipeline.enable_model_cpu_offload()
# num_inference_steps = 50

# def interrupt_callback(pipeline, i, t, callback_kwargs):
#     stop_idx = 10
#     if i == stop_idx:
#         pipeline._interrupt = True

#     return callback_kwargs

# img = pipeline(
#     "A photo of a cat.",
#     # height = 128,
#     # width = 128,
#     num_inference_steps=num_inference_steps,
#     # callback_on_step_end=interrupt_callback,
# ).images[0]
# img.save("./test_img2.jpg")