import numpy as np
from torchvision import transforms
from .randaugment import RandAugmentMC
import math
from PIL import Image

def x_u_split_known_novel(labels, lbl_percent, no_classes, lbl_set, unlbl_set, val_percent=1):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    val_idx = []
    for i in range(no_classes):
        idx = np.where(labels == i)[0]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
        # n_val_sample = max(int(len(idx)*(val_percent/100)), 1)
        np.random.shuffle(idx)
        if i in lbl_set:
            labeled_idx.extend(idx[:n_lbl_sample])
            # unlabeled_idx.extend(idx[n_lbl_sample:-n_val_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:])
            # val_idx.extend(idx[-n_val_sample:])
        elif i in unlbl_set:
            # unlabeled_idx.extend(idx[:-n_val_sample])
            unlabeled_idx.extend(idx)
            # val_idx.extend(idx[-n_val_sample:])
    return labeled_idx, unlabeled_idx, val_idx


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
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)