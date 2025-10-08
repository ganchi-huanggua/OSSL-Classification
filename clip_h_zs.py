import os
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载CLIP模型和预处理器（兼容open-clip-torch 3.2.0）
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-H-14",
    pretrained="laion2B-s32B-b79K",
    cache_dir="./pretrained",
    device=device
)
tokenizer = open_clip.get_tokenizer("ViT-H-14")

# 设置评估模式
model.eval()

# CUB200数据集类
class CUB200Dataset(Dataset):
    def __init__(self, root, split='test', transform=None):
        self.root = os.path.join(root, 'CUB_200_2011')
        self.split = split
        self.transform = transform
        
        # 加载图像路径
        self.image_paths = {}
        with open(os.path.join(self.root, 'images.txt'), 'r') as f:
            for line in f:
                img_id, path = line.strip().split()
                self.image_paths[img_id] = path
        
        # 加载训练/测试划分
        self.split_ids = []
        split_file = 'train_test_split.txt'
        with open(os.path.join(self.root, split_file), 'r') as f:
            for line in f:
                img_id, is_train = line.strip().split()
                if (split == 'train' and is_train == '1') or (split == 'test' and is_train == '0'):
                    self.split_ids.append(img_id)
        
        # 加载标签名称
        self.class_names = {}
        with open(os.path.join(self.root, 'classes.txt'), 'r') as f:
            for line in f:
                class_id, name = line.strip().split()
                # 提取鸟类名称（CUB200格式为"classxxx.鸟类名称"）
                bird_name = ' '.join(name.split('.')[1:])
                self.class_names[class_id] = bird_name
        
        # 加载图像标签
        self.image_labels = {}
        with open(os.path.join(self.root, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.image_labels[img_id] = int(class_id) - 1  # 转为0基索引

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, idx):
        img_id = self.split_ids[idx]
        img_path = os.path.join(self.root, 'images', self.image_paths[img_id])
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[img_id]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 生成类别提示词
def get_cub_prompts(class_names):
    """为CUB200鸟类生成提示词模板"""
    prompts = []
    for name in class_names:
        # 使用多种提示词模板提高准确性
        prompts.append(f"a photo of a {name} bird")
        prompts.append(f"an image of a {name}")
        prompts.append(f"picture of a {name} bird")
        prompts.append(f"{name}")
    return prompts

# 加载测试数据集
def load_cub_dataset(data_root, batch_size=32):
    test_transform = preprocess_val  # 使用模型提供的验证集预处理
    
    # 先加载训练集获取类别名称（保持类别顺序一致）
    train_dataset = CUB200Dataset(data_root, split='train', transform=None)
    class_names = [train_dataset.class_names[str(i+1)] for i in range(200)]
    
    # 加载测试集
    test_dataset = CUB200Dataset(data_root, split='test', transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    return test_loader, class_names

# 计算zero-shot准确率
def zero_shot_accuracy(data_root, batch_size=32):
    # 加载数据和类别名称
    test_loader, class_names = load_cub_dataset(data_root, batch_size)
    print(f"类别数量: {len(class_names)}")
    print(f"测试集样本数量: {len(test_loader.dataset)}")
    
    # 生成并编码提示词
    prompts = get_cub_prompts(class_names)
    text_input = tokenizer(prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # 归一化
    
    # 每个类别有多个提示词，取平均
    num_templates = 4  # 每个类别使用的提示词数量
    class_text_features = text_features.view(-1, num_templates, text_features.shape[-1])
    class_text_features = class_text_features.mean(dim=1)  # 平均提示词特征
    class_text_features /= class_text_features.norm(dim=-1, keepdim=True)  # 重新归一化
    
    # 测试循环
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="测试中"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取图像特征
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化
            
            # 计算相似度
            logits = (image_features @ class_text_features.T) * 100.0  # 缩放因子
            
            # 预测
            _, predicted = torch.max(logits, 1)
            
            # 统计
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy, class_names

if __name__ == "__main__":
    # 设置CUB200数据集根目录（请替换为你的实际路径）
    CUB_DATA_ROOT = "/home/lhz/data"  # 应包含CUB_200_2011文件夹
    
    # 运行zero-shot测试
    acc, classes = zero_shot_accuracy(CUB_DATA_ROOT, batch_size=32)
    
    # 输出结果
    print(f"\nCLIP ViT-H-14 在CUB200测试集上的zero-shot准确率: {acc:.2f}%")
    print(f"测试样本总数: {len(classes)*50} (每个类别约50个测试样本)")
    