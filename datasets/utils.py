import numpy as np
from torchvision import transforms
from .randaugment import RandAugmentMC
import math
import json
import logging
from torch.utils.data import Dataset

def x_u_split_known_novel(labels, lbl_percent, no_classes, lbl_set, unlbl_set, val_percent=False):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    val_idx = []
    for i in range(no_classes):
        idx = np.where(labels == i)[0]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
        if val_percent:
            n_val_sample = max(int(len(idx)*(20/100)), 1)
        np.random.shuffle(idx)
        if i in lbl_set:
            if val_percent:
                labeled_idx.extend(idx[:n_lbl_sample])
                unlabeled_idx.extend(idx[n_lbl_sample:])
                val_idx.extend(idx[-n_val_sample:])
            else:
                labeled_idx.extend(idx[:n_lbl_sample])
                unlabeled_idx.extend(idx[n_lbl_sample:])
        elif i in unlbl_set:
            if val_percent:
                unlabeled_idx.extend(idx)
                val_idx.extend(idx[-n_val_sample:])
            else:
                unlabeled_idx.extend(idx)
            # 从unlabeled_idx中取出最后n个样本加入到labeled_idx中
            # n = 10  # 设置要移动的样本数量
            # if len(unlabeled_idx) > n:
            #     last_n = unlabeled_idx[-n:]
            #     unlabeled_idx = unlabeled_idx[:-n]
            #     labeled_idx.extend(last_n)
    logging.info(f"{len(labeled_idx)}, {len(unlabeled_idx)}")
    logging.info(f"{set(labels[labeled_idx])}, {set(labels[unlabeled_idx])}")
    return labeled_idx, unlabeled_idx, val_idx


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


class TransformWS32(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((int(math.floor(32 / 0.875)), int(math.floor(32 / 0.875)))),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((int(math.floor(32 / 0.875)), int(math.floor(32 / 0.875)))),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformWS64(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((int(math.floor(64 / 0.875)), int(math.floor(64 / 0.875)))),
            transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect')
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformWS224(object):
    def __init__(self, mean, std):
        self.origin = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
            transforms.CenterCrop(224),
        ])
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((int(math.floor(224 / 0.875)), int(math.floor(224 / 0.875)))),
            transforms.RandomResizedCrop(224),
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((int(math.floor(224 / 0.875)), int(math.floor(224 / 0.875)))),
            transforms.RandomResizedCrop(224),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        origin = self.origin(x)
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(origin), self.normalize(weak), self.normalize(strong)
    
    
class PseudoLabelDataset(Dataset):
    def __init__(self, original_dataset, index_to_pseudo_label):
        """
        Args:
            original_dataset: 原始无标签数据集（unlbl_loader.dataset）
            index_to_pseudo_label: 样本索引→伪标签的映射字典
        """
        self.original_dataset = original_dataset
        self.index_to_pseudo_label = index_to_pseudo_label
        # 筛选出原始数据集中“被选中的样本”（避免加载无关样本）
        self.selected_indices = list(index_to_pseudo_label.keys())

    def __len__(self):
        # 数据集长度 = 被选中的样本数量
        return len(self.selected_indices)

    def __getitem__(self, idx_in_subset):
        """
        重写加载逻辑：返回“原始样本数据 + 伪标签”
        Args:
            idx_in_subset: 自定义数据集内的索引（0 ~ len(self)-1）
        Returns:
            与原始数据集格式一致，但标签替换为伪标签
        """
        # 1. 根据自定义数据集的索引，获取原始数据集中的样本索引
        original_idx = self.selected_indices[idx_in_subset]
        
        # 2. 加载原始样本（原始数据 + 原始标签，原始标签会被替换）
        # 注意：原始数据集的__getitem__返回格式需与这里匹配（根据你的数据格式调整）
        # 假设原始数据集返回：(inputs_u, inputs_u_w, inputs_u_s), original_label, original_idx
        original_sample = self.original_dataset[original_idx]
        
        # 3. 提取原始样本的“数据部分”和“原始索引”，替换标签为伪标签
        # （根据你的原始数据格式调整解构方式，这里匹配你之前的data_unlbl格式）
        (inputs_u, inputs_u_w, inputs_u_s), _, original_idx = original_sample
        pseudo_label = self.index_to_pseudo_label[original_idx]  # 获取伪标签
        
        # 4. 返回与原始格式一致的样本（数据部分不变，标签改为伪标签）
        return (inputs_u, inputs_u_w, inputs_u_s), pseudo_label, idx_in_subset