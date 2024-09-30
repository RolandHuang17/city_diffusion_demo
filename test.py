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