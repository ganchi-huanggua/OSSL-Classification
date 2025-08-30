import numpy as np
from torchvision import transforms
from .randaugment import RandAugmentMC
import math
from PIL import Image
import os.path as osp
import json
import logging

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
                unlabeled_idx.extend(idx[n_lbl_sample:-n_val_sample])
                val_idx.extend(idx[-n_val_sample:])
            else:
                labeled_idx.extend(idx[:n_lbl_sample])
                unlabeled_idx.extend(idx[n_lbl_sample:])
        elif i in unlbl_set:
            if val_percent:
                unlabeled_idx.extend(idx[:-n_val_sample])
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
    
