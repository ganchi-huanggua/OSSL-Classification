import numpy as np
from torchvision import datasets, transforms
import pickle
import os
from PIL import Image
from .utils import x_u_split_known_novel, TransformWS224, TransformWS32, TransformWS64
import logging
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataset224(args):
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

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
    base_dataset_targets = np.array(base_dataset.imgs)
    base_dataset_targets = base_dataset_targets[:, 1]
    base_dataset_targets = list(map(int, base_dataset_targets.tolist()))
    if not os.path.exists(args.ssl_indexes):
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict, f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
    logging.info(f"{len(train_labeled_idxs)}, {len(train_unlabeled_idxs)}")
    logging.info(f"{set(np.array(base_dataset_targets)[train_labeled_idxs])}, {set(np.array(base_dataset_targets)[train_unlabeled_idxs])}")
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
    train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, transform=TransformWS224(mean=imgnet_mean, std=imgnet_std))
    train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformWS224(mean=imgnet_mean, std=imgnet_std))
    test_dataset_known = GenericTEST(os.path.join(args.data_root, 'val'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(0, args.no_known)))
    test_dataset_novel = GenericTEST(os.path.join(args.data_root, 'val'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = GenericTEST(os.path.join(args.data_root, 'val'), no_class=args.no_class, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset_known, test_dataset_novel, test_dataset_all, ['robin', 'water_ouzel', 
        'box_turtle', 'sea_snake', 'diamondback', 'sidewinder', 'scorpion', 'goose', 'tusker', 'American_coot', 'oystercatcher', 
        'albatross', 'toy_terrier', 'bluetick', 'Staffordshire_bullterrier', 'Border_terrier', 'Norfolk_terrier', 'cairn', 'giant_schnauzer', 
        'Scotch_terrier', 'flat-coated_retriever', 'Irish_setter', 'schipperke', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Doberman', 
        'dalmatian', 'coyote', 'Arctic_fox', 'grey_fox', 'cougar', 'leopard', 'American_black_bear', 'ringlet', 'wood_rabbit', 'guinea_pig', 
        'guenon', 'proboscis_monkey', 'analog_clock', 'ashcan', 'bicycle-built-for-two', 'broom', 'bucket', 'computer_keyboard', 'cowboy_hat', 
        'crash_helmet', 'dam', 'dumbbell', 'electric_guitar', 'envelope', 'file', 'gown', 'hand_blower', 'hatchet', 'honeycomb', 'knee_pad', 
        'lawn_mower', 'maillot', 'manhole_cover', 'maze', 'microphone', 'mitten', 'neck_brace', 'obelisk', 'oboe', 'organ', 'pickelhaube', 
        'picket_fence', 'plane', 'planetarium', 'pop_bottle', 'printer', 'purse', 'recreational_vehicle', 'shoe_shop', 'shower_curtain', 
        'sleeping_bag', 'steel_arch_bridge', 'stole', 'stretcher', 'stupa', 'table_lamp', 'thresher', 'tobacco_shop', 'totem_pole', 'trimaran', 
        'unicycle', 'upright', 'vending_machine', 'washer', 'Windsor_tie', 'wing', 'wreck', 'guacamole', 'trifle', 'bagel', 'mashed_potato', 
        'banana', 'rapeseed']


class GenericSSL(datasets.ImageFolder):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class GenericTEST(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, labeled_set=None, no_class=100):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(no_class):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
