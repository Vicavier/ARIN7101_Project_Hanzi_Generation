from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms


class HanziDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集所在的文件夹路径
            transform (callable, optional): 应用于每个样本的图像预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

        # 定义汉字到数字的映射
        self.hanzi_list = ['艾', '达', '访', '谷', '禾', '臼', '西', '羽', '长']  # 汉字列表
        self.hanzi_to_num = {char: idx for idx, char in enumerate(self.hanzi_list)}  # 创建映射

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取文件名
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 打开图像并转换为灰度图
        image = Image.open(img_path).convert('L')  # 使用 'L' 模式表示单通道灰度图
        
        # 从文件名的第一个字符获取标签（假设第一个字符是汉字）
        label_char = img_name[0]  # 获取字符
        
        # 将汉字转换为对应的数字标签
        label_num = self.hanzi_to_num[label_char]  # 从字典中查找对应的数字
        
        # 将标签转换为 PyTorch tensor
        label_tensor = torch.tensor(label_num, dtype=torch.long)  # 标签为 long 类型的 tensor
        
        # 应用预处理变换
        if self.transform:
            image = self.transform(image)
        
        return image, label_tensor
    

def create_hanzi_dataloaders(root_dir, batch_size, preprocess):
    
    # 创建自定义数据集
    dataset = HanziDataset(root_dir=root_dir, transform=preprocess)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader