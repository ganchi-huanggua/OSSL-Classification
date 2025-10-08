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


def get_stanfordcars(args):
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
    filepath = os.path.join("/home/lhz/data/stanford_cars", "split_zhou_StanfordCars.json")
    image_dir = f"/home/lhz/data/stanford_cars"
    tracker = defaultdict(list)
    lab2cname = dict()
    def _convert(items):
        for impath, label, classname in items:
            impath = os.path.join(image_dir, impath)
            label = int(label)
            tracker[label].append(impath)
            if label not in lab2cname.keys():
                lab2cname[label] = classname
            
    with open(filepath, "r") as f:
        split = json.load(f)
    _convert(split["train"])
    _convert(split["val"])
    _convert(split["test"])
    
    # cnames = [lab2cname[k] for k in sorted(lab2cname, key=lambda x: int(x))]
    PATH_TO_PROMPTS = f'gpt3_prompts/cleaned_CuPL_prompts_stanford_cars.json'
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)
    cnames = {}
    for item in gpt3_prompts.items():
        cnames[item[0]] = item[1]
    # train_classes_textnames = [textnames[i] for i in args.train_classes]
    # unlabeled_classes_textnames = [textnames[i] for i in args.unlabeled_classes]
    # textnames = train_classes_textnames + unlabeled_classes_textnames
    
    data, targets = [], []
    for label, impaths in tracker.items():
        data.extend(impaths)
        targets.extend([label] * len(impaths))
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
    logging.info(f"{len(train_labeled_idxs)}, {len(train_unlabeled_idxs)}")
    logging.info(f"{set(np.array(targets)[train_labeled_idxs])}, {set(np.array(targets)[train_unlabeled_idxs])}")
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
    train_labeled_dataset = GenericSSL(data, targets, train_labeled_idxs, transform=TransformWS224(imgnet_mean, imgnet_std))
    train_unlabeled_dataset = GenericSSL(data, targets, train_unlabeled_idxs, transform=TransformWS224(imgnet_mean, imgnet_std))
    
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
        return img, target, index


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
    
    