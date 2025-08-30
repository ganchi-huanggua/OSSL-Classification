import numpy as np
from torchvision import datasets, transforms
import pickle
from PIL import Image
import os
from .utils import x_u_split_known_novel, TransformWS224, TransformWS32, TransformWS64

cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

def get_cifar10(args):
    # augmentations
    # transform_labeled = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=32,
    #                           padding=int(32*0.125),
    #                           padding_mode='reflect'),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    # ])
    if args.img_size == 224:
        transform_val = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
            transforms.CenterCrop(224),  # 实际没必要crop，但写上更标准
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean,  # CIFAR-10官方均值
                                std=cifar10_std)
        ])
    else:
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
    base_dataset = datasets.CIFAR10(args.data_root, train=True, download=True)
    # 从CIFAR10数据集中获取类别名称
    classes = base_dataset.classes
    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'cifar10_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    # import ipdb;ipdb.set_trace()
    # if len(train_unlabeled_idxs) > len(train_labeled_idxs):
    #     exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
    #     train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

    #     if len(train_labeled_idxs) < len(train_unlabeled_idxs):
    #         diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
    #         train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
    #     else:
    #         assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    if args.img_size == 224:
        train_labeled_dataset = CIFAR10SSL(args.data_root, train_labeled_idxs, train=True, transform=TransformWS224(mean=cifar10_mean, std=cifar10_std))
        train_unlabeled_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS224(mean=cifar10_mean, std=cifar10_std))
    else:
        train_labeled_dataset = CIFAR10SSL(args.data_root, train_labeled_idxs, train=True, transform=TransformWS32(mean=cifar10_mean, std=cifar10_std))
        train_unlabeled_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS32(mean=cifar10_mean, std=cifar10_std))
    # train_pl_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val)
    test_dataset_known = CIFAR10SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(0,args.no_known)))
    test_dataset_novel = CIFAR10SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = CIFAR10SSL_TEST(args.data_root, train=False, transform=transform_val, download=False)
    
    # test_dataset_known = CIFAR10SSL_trans(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val, download=False, labeled_set=list(range(0,args.no_known)))
    # test_dataset_novel = CIFAR10SSL_trans(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    # test_dataset_all = CIFAR10SSL_trans(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val, download=False)


    return train_labeled_dataset, train_unlabeled_dataset, test_dataset_known, test_dataset_novel, test_dataset_all, classes


def get_cifar100(args):
    # augmentations
    # transform_labeled = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=32,
    #                           padding=int(32*0.125),
    #                           padding_mode='reflect'),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    if args.img_size == 224:
        transform_val = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 保持结构更好
            transforms.CenterCrop(224),  # 实际没必要crop，但写上更标准
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean,  # CIFAR-10官方均值
                                std=cifar100_std)
        ])
    else:
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
    # 从CIFAR100数据集中获取类别名称
    classes = base_dataset.classes
    if not os.path.exists(args.ssl_indexes):
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'cifar100_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict, f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # # balance the labeled and unlabeled data
    # if len(train_unlabeled_idxs) > len(train_labeled_idxs):
    #     exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
    #     train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

    #     if len(train_labeled_idxs) < len(train_unlabeled_idxs):
    #         diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
    #         train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
    #     else:
    #         assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    if args.img_size == 224:
        train_labeled_dataset = CIFAR100SSL(args.data_root, train_labeled_idxs, train=True, transform=TransformWS224(mean=cifar100_mean, std=cifar100_std))
        train_unlabeled_dataset = CIFAR100SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS224(mean=cifar100_mean, std=cifar100_std))
    else:
        train_labeled_dataset = CIFAR100SSL(args.data_root, train_labeled_idxs, train=True, transform=TransformWS32(mean=cifar100_mean, std=cifar100_std))
        train_unlabeled_dataset = CIFAR100SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS32(mean=cifar100_mean, std=cifar100_std))
    
    test_dataset_known = CIFAR100SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(0, args.no_known)))
    test_dataset_novel = CIFAR100SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = CIFAR100SSL_TEST(args.data_root, train=False, transform=transform_val, download=False)

    # test_dataset_known = CIFAR100SSL_trans(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val, download=False, labeled_set=list(range(0,args.no_known)))
    # test_dataset_novel = CIFAR100SSL_trans(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    # test_dataset_all = CIFAR100SSL_trans(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val, download=False)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset_known, test_dataset_novel, test_dataset_all, classes


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR10SSL_TEST(datasets.CIFAR10):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    

class CIFAR100SSL_TEST(datasets.CIFAR100):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(100):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target