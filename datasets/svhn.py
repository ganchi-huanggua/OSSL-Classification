import numpy as np
from torchvision import datasets, transforms
import pickle
from PIL import Image
import os
from .utils import x_u_split_known_novel, TransformWS224, TransformWS32, TransformWS64

normal_mean, normal_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def get_svhn(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        base_dataset = datasets.SVHN(args.data_root, split='train', download=True)
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset.labels, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'svhn_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = SVHNSSL(args.data_root, train_labeled_idxs, split='train', transform=TransformWS32(mean=normal_mean, std=normal_std))
    train_unlabeled_dataset = SVHNSSL(args.data_root, train_unlabeled_idxs, split='train', transform=TransformWS32(mean=normal_mean, std=normal_std))
    # train_pl_dataset = SVHNSSL(args.data_root, train_unlabeled_idxs, split='train', transform=transform_val)
    test_dataset_known = SVHNSSL_TEST(args.data_root, split='test', transform=transform_val, download=True, labeled_set=list(range(0,args.no_known)))
    test_dataset_novel = SVHNSSL_TEST(args.data_root, split='test', transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = SVHNSSL_TEST(args.data_root, split='test', transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.labels = np.array(self.labels)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.labels))

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.moveaxis(img, 0, -1))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SVHNSSL_TEST(datasets.SVHN):
    def __init__(self, root, split='test',
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.labels = np.array(self.labels)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.labels == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.moveaxis(img, 0, -1))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target