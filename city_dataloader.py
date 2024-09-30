from torch.utils.data import DataLoader
import torchvision
from four_city_dataset import train_dataset


# 设置批量大小和 DataLoader 的其他参数
batch_size = 32
shuffle = True
num_workers = 0

# 为拼接的数据集创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# 示例：遍历 DataLoader，打印每个批次的信息
for batch_idx, (images, labels) in enumerate(train_loader):
    print(images.shape)
    print(f"批次 {batch_idx + 1}")
    print(f"该批次中的图片数量: {len(images)}")
    print(f"标签: {labels}")
