from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision
import os

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = torchvision.transforms.ToTensor()  # 定义将 PIL 图像转换为 Tensor 的转换

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)

        img = self.transform(img)

        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "buildings"
beijing_label_dir = "Beijing_image"
guangzhou_label_dir = "Guangzhou_image"
hongkong_label_dir = "Hongkong_image"
wuhan_label_dir = "Wuhan_image"

beijing_dataset = MyDataset(root_dir,beijing_label_dir)
guangzhou_dataset = MyDataset(root_dir,guangzhou_label_dir)
hongkong_dataset = MyDataset(root_dir,hongkong_label_dir)
wuhan_dataset = MyDataset(root_dir,wuhan_label_dir)

train_dataset = ConcatDataset([beijing_dataset, guangzhou_dataset, hongkong_dataset, wuhan_dataset])



