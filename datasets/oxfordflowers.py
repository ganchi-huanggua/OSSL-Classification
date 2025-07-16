import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
from .utils import x_u_split_known_novel, TransformWS224, TransformWS32, TransformWS64, read_json, Datum
from collections import defaultdict
import random
from scipy.io import loadmat

imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_flowers(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
    ])
    
    tracker = defaultdict(list)
    label_file = loadmat("/home/lhz/data/oxford_flowers/imagelabels.mat")["labels"][0]
    for i, label in enumerate(label_file):
        imname = f"image_{str(i + 1).zfill(5)}.jpg"
        impath = os.path.join("/home/lhz/data/oxford_flowers/jpg", imname)
        label = int(label)
        tracker[label].append(impath)
        
    lab2cname = read_json("/home/lhz/data/oxford_flowers/cat_to_name.json")
    cnames = [lab2cname[k] for k in sorted(lab2cname, key=lambda x: int(x))]
    data, targets = [], []
    for label, impaths in tracker.items():
        data.extend(impaths)
        targets.extend([label - 1] * len(impaths))
            # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split_known_novel(
            targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)), val_percent=True)
        f = open(os.path.join(args.split_root, f'{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': val_idxs}
        pickle.dump(label_unlabel_dict, f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
        val_idxs = label_unlabel_dict['val_idx']

    # balance the labeled and unlabeled data
    # if len(train_unlabeled_idxs) > len(train_labeled_idxs):
    #     exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
    #     train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

    #     if len(train_labeled_idxs) < len(train_unlabeled_idxs):
    #         diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
    #         train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
    #     else:
    #         assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = GenericSSL(data, targets, train_labeled_idxs, transform=TransformWS224(mean=imgnet_mean, std=imgnet_std))
    train_unlabeled_dataset = GenericSSL(data, targets, train_unlabeled_idxs, transform=TransformWS224(mean=imgnet_mean, std=imgnet_std))
    
    test_dataset_known = GenericTEST(data, targets, val_idxs, transform=transform_val, labeled_set=list(range(0, args.no_known)))
    test_dataset_novel = GenericTEST(data, targets, val_idxs, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = GenericTEST(data, targets, val_idxs, transform=transform_val)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset_known, test_dataset_novel, test_dataset_all, cnames


class GenericSSL(Dataset):
    def __init__(self, data, targets, index_list, transform=None):
        """
        Semi-Supervised Dataset: 用于训练
        Args:
            data: 所有样本路径 list[str]
            targets: 所有样本标签 list[int]
            index_list: 当前 split 的索引（labeled 或 unlabeled）
            transform: 图像变换
        """
        self.data = [data[i] for i in index_list]
        self.targets = [targets[i] for i in index_list]
        self.transform = transform
        self.index_list = index_list  # 方便 debug 或追踪

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        target = self.targets[index]
        return img, target, self.index_list[index]


class GenericTEST(Dataset):
    def __init__(self, data, targets, val_idxs, transform=None, labeled_set=None):
        """
        Args:
            data: list[str] → 图片路径
            targets: list[int] → 标签
            val_idxs: list[int] → 验证集索引
            transform: torchvision transforms
            labeled_set: list[int] → 如果指定，只保留 val_idxs 中标签属于 labeled_set 的样本
        """
        self.data = []
        self.targets = []
        self.indices = []
        self.transform = transform

        for idx in val_idxs:
            label = targets[idx]
            if labeled_set is None or label in labeled_set:
                self.data.append(data[idx])
                self.targets.append(label)
                self.indices.append(idx)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[index]
    
    