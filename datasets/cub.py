import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
from .utils import x_u_split_known_novel, TransformWS224, TransformWS32, TransformWS64, read_json
from collections import defaultdict
import json
from scipy.io import loadmat
import logging
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
clip_mean, clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

def get_cub(args):
    # augmentations
    # transform_labeled = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
    # ])

    transform_val = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
    ])

    # lab2cname = read_json("/home/lhz/data/oxford_flowers/cat_to_name.json")
    # cnames = [lab2cname[k] for k in sorted(lab2cname, key=lambda x: int(x))]
    PATH_TO_PROMPTS = f'gpt3_prompts/cleaned_CuPL_prompts_cub.json'
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)
    cnames = {}
    for item in gpt3_prompts.items():
        cnames[item[0]] = item[1]
    # train_classes_textnames = [textnames[i] for i in args.train_classes]
    # unlabeled_classes_textnames = [textnames[i] for i in args.unlabeled_classes]
    # textnames = train_classes_textnames + unlabeled_classes_textnames
    dataset_dir = "/home/lhz/data/CUB_200_2011"
    # dset = datasets.ImageFolder(dataset_dir)

    split_file = os.path.join(dataset_dir, 'train_test_split.txt')
    train_indices = []
    test_indices = []
    
    with open(split_file, 'r') as f:
        for line in f:
            img_id, is_train = line.strip().split()
            # CUB的ID是从1开始的，转换为0基索引
            idx = int(img_id) - 1
            if is_train == '1':
                train_indices.append(idx)
            else:
                test_indices.append(idx)
    
    image_paths = {}
    with open(os.path.join(dataset_dir, 'images.txt'), 'r') as f:
        for line in f:
            img_id, path = line.strip().split()
            image_paths[int(img_id)-1] = path  # 转换为0基索引
    
    # 加载图像标签
    image_labels = {}
    with open(os.path.join(dataset_dir, 'image_class_labels.txt'), 'r') as f:
        for line in f:
            img_id, class_id = line.strip().split()
            # 转换为0基索引（CUB原始类别从1开始）
            image_labels[int(img_id)-1] = int(class_id) - 1   
    
    all_indices = np.arange(len(image_paths))
    data = [os.path.join(dataset_dir, "images", image_paths[idx]) for idx in all_indices]
    targets = [image_labels[idx] for idx in all_indices]
    
    train_data = [data[idx] for idx in train_indices]
    train_targets = [targets[idx] for idx in train_indices]

    if not os.path.exists(args.ssl_indexes):
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split_known_novel(
            train_targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))
        f = open(os.path.join(args.split_root, f'{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': val_idxs}
        pickle.dump(label_unlabel_dict, f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
        val_idxs = label_unlabel_dict['val_idx']
        
    logging.info(f"{len(train_labeled_idxs)}, {len(train_unlabeled_idxs)}")
    logging.info(f"{set(np.array(train_targets)[train_labeled_idxs])}, {set(np.array(train_targets)[train_unlabeled_idxs])}")
    # generate datasets
    train_labeled_dataset = GenericSSL(train_data, train_targets, train_labeled_idxs, transform=TransformWS224(imgnet_mean, imgnet_std))
    train_unlabeled_dataset = GenericSSL(train_data, train_targets, train_unlabeled_idxs, transform=TransformWS224(imgnet_mean, imgnet_std))
    
    test_data = [data[idx] for idx in test_indices]
    test_targets = [targets[idx] for idx in test_indices]
    test_dataset_known = GenericTEST(test_data, test_targets, transform=transform_val, labeled_set=list(range(0, args.no_known)))
    test_dataset_novel = GenericTEST(test_data, test_targets, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = GenericTEST(test_data, test_targets, transform=transform_val)
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
        return img, target, index


class GenericTEST(Dataset):
    def __init__(self, data, targets, transform=None, labeled_set=None):
        """
        Args:
            data: list[str] → 图片路径（已经是测试集）
            targets: list[int] → 标签（已经是测试集）
            transform: torchvision transforms
            labeled_set: list[int] or None → 已知类的标签集合
                         如果为 None，保留所有样本（test_dataset_all）
                         如果为 list，只保留标签在 labeled_set 中的样本（已知类）
                         新颖类则传入 novel_classes 列表
        """
        self.transform = transform

        if labeled_set is None:
            # 保留所有样本
            self.data = data
            self.targets = targets
        else:
            # 过滤出属于 labeled_set 的样本
            labeled_set = set(labeled_set)
            self.data = []
            self.targets = []
            for img_path, label in zip(data, targets):
                if label in labeled_set:
                    self.data.append(img_path)
                    self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[index]
    
    