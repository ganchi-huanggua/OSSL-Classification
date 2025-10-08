from .cifar import get_cifar10, get_cifar100
from .svhn import get_svhn
from .tinyimagenet import get_tinyimagenet
from .imagenet100 import get_dataset224
from .oxfordflowers import get_flowers
from .oxfordpets import get_oxfordpets
from .stanfordcars import get_stanfordcars
from .cub import get_cub

def get_dataset(args):
    if args.dataset == 'cifar10':
        return get_cifar10(args)
    elif args.dataset == 'cifar100':
        return get_cifar100(args)
    elif args.dataset == 'svhn':
        return get_svhn(args)
    elif args.dataset == 'tinyimagenet':
        return get_tinyimagenet(args)
    elif args.dataset == 'oxfordflowers':
        return get_flowers(args)
    elif args.dataset == 'stanfordcars':
        return get_stanfordcars(args)
    elif args.dataset == 'oxfordpets':
        return get_oxfordpets(args)
    elif args.dataset == 'cub':
        return get_cub(args) 
    elif args.dataset in ['aircraft', 'imagenet100', 'herbarium']:
        return get_dataset224(args)


