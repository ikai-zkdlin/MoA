# --------------------------------------------------------
# References: https://github.com/ShoufaChen/AdaptFormer
# --------------------------------------------------------
import os
from typing import Any, Callable, Optional
from util.crop import RandomResizedCrop
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .json_dataset import JSONDataset
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data import create_transform
# from datasets import load_dataset
def build_image_dataset(args):
    # linear probe: weak augmentation
    if os.path.basename(args.finetune).startswith('jx_'):
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD
    elif os.path.basename(args.finetune).startswith('mae_pretrain_vit'):
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
    elif os.path.basename(args.finetune).startswith('swin_base_patch4'):
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
    else:
        raise ValueError(os.path.basename(args.finetune))
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])

    if args.dataset == 'imagenet':
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        nb_classes = 1000
    elif args.dataset == 'dtd':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'dtd/split_zhou_DescribableTextures.json'),
                                    os.path.join(args.data_path, 'dtd/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'dtd/split_zhou_DescribableTextures.json'),
                                  os.path.join(args.data_path, 'dtd/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 47
    elif args.dataset == 'oxford_pets':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'oxford_pets/split_zhou_OxfordPets.json'),
                                    os.path.join(args.data_path, 'oxford_pets/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'oxford_pets/split_zhou_OxfordPets.json'),
                                  os.path.join(args.data_path, 'oxford_pets/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 37
    elif args.dataset == 'RESISC45':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'resisc45/resisc45_json.json'),
                                    os.path.join(args.data_path, 'resisc45/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'resisc45/resisc45_json.json'),
                                  os.path.join(args.data_path, 'resisc45/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 45
    elif args.dataset == 'ucf101':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'ucf101/split_zhou_UCF101.json'),
                                    os.path.join(args.data_path, 'ucf101/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'ucf101/split_zhou_UCF101.json'),
                                  os.path.join(args.data_path, 'ucf101/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 101
    elif args.dataset == 'sun397':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'sun397/split_zhou_SUN397.json'),
                                    os.path.join(args.data_path, 'sun397/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'sun397/split_zhou_SUN397.json'),
                                  os.path.join(args.data_path, 'sun397/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 397
    elif args.dataset == 'eurosat':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'eurosat/split_zhou_EuroSAT.json'),
                                    os.path.join(args.data_path, 'eurosat/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'eurosat/split_zhou_EuroSAT.json'),
                                  os.path.join(args.data_path, 'eurosat/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 10
    elif args.dataset == 'caltech101':
        dataset_train = JSONDataset(os.path.join(args.data_path, 'caltech-101/split_zhou_Caltech101.json'),
                                    os.path.join(args.data_path, 'caltech-101/images'),
                                    split='train',
                                    transforms=transform_train)
        dataset_val = JSONDataset(os.path.join(args.data_path, 'caltech-101/split_zhou_Caltech101.json'),
                                  os.path.join(args.data_path, 'caltech-101/images'),
                                  split='test',
                                  transforms=transform_val)
        nb_classes = 102
    elif args.dataset == 'fgvc_aircraft':
        dataset_train = datasets.FGVCAircraft(os.path.join(args.data_path, 'FGVCAircraft'), 
                                              transform=transform_train, annotation_level= 'variant',split='train',download=True )
        dataset_val = datasets.FGVCAircraft(os.path.join(args.data_path, 'FGVCAircraft'), 
                                            transform=transform_val, annotation_level= 'variant',split='test',download=True )
        nb_classes = 100
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'), 
                                         transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR10(os.path.join(args.data_path, 'cifar10'), 
                                       transform=transform_val, train=False, download=True)
        nb_classes = 10
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), 
                                          transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), 
                                        transform=transform_val, train=False, download=True)
        nb_classes = 100
    elif args.dataset == 'oxford_flowers':
        from .flowers102 import Flowers102
        dataset_train = Flowers102(os.path.join(args.data_path, 'flowers102'), split='train', 
                                   transform=transform_train, download=True)
        dataset_val = Flowers102(os.path.join(args.data_path, 'flowers102'), split='test', 
                                 transform=transform_val, download=True)
        nb_classes = 102
    elif args.dataset == 'svhn':
        from torchvision.datasets import SVHN
        dataset_train = SVHN(os.path.join(args.data_path, 'svhn'), split='train', 
                             transform=transform_train, download=True)
        dataset_val = SVHN(os.path.join(args.data_path, 'svhn'), split='test', 
                           transform=transform_val, download=True)
        nb_classes = 10
    elif args.dataset == 'food-101':
        from .food101 import Food101
        dataset_train = Food101(os.path.join(args.data_path, 'food101'), split='train', 
                                transform=transform_train, download=True)
        dataset_val = Food101(os.path.join(args.data_path, 'food101'), split='test', 
                              transform=transform_val, download=True)
        nb_classes = 101
    elif args.dataset == 'FER2013':
        dataset_train = datasets.FER2013(os.path.join(args.data_path, 'fer2013'), 
                                         transform=transform_train, split='train', download=True)
        dataset_val = datasets.FER2013(os.path.join(args.data_path, 'fer2013'), 
                                       transform=transform_val, split='test', download=True)
        nb_classes = 7
    elif args.dataset == 'GTSRB':
        dataset_train = datasets.GTSRB(os.path.join(args.data_path, 'GTSRB'), 
                                       transform=transform_train, split='train', download=True)
        dataset_val = datasets.GTSRB(os.path.join(args.data_path, 'GTSRB'), 
                                     transform=transform_val, split='test', download=True)
        nb_classes = 43
    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val, nb_classes

class general_dataset_few_shot(ImageFolder):
    def __init__(self, root, dataset, target_root=None, target_dataset=None, train=True, 
                 transform=None, target_transform=None, mode=None, shot=2, seed=0, **kwargs):
        self.root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        self.dataset = dataset
        self.target_root = target_root
        self.target_dataset = target_dataset
        
        if 'imagenet' in root:
            train_list_path = os.path.join(self.root, 'annotations/train_meta.list.num_shot_' + str(shot) + '.seed_' + str(seed))
        else:
            train_list_path = os.path.join(self.root, str(dataset) + '/annotations/train_meta.list.num_shot_' + str(shot) + '.seed_' + str(seed))
        
        if 'imagenet' in root:
            test_list_path = os.path.join(self.target_root, str(target_dataset) + '/annotations/val_meta.list')
        else:
            test_list_path = os.path.join(self.root, str(dataset) + '/annotations/test_meta.list')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'imagenet' == self.dataset:
                        self.samples.append((os.path.join(root + 'data/train', img_name), label))
                    else:
                        self.samples.append((os.path.join(root + str(dataset) + '/images', img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    # official validation
                    if 'imagenet' == self.target_dataset:
                        self.samples.append((os.path.join(root + 'data/', img_name), label))
                    else:
                        self.samples.append((os.path.join(target_root + str(target_dataset) + '/', img_name), label))



def build_dataset_few_shot(args):
    if os.path.basename(args.finetune).startswith('jx_'):
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD
    elif os.path.basename(args.finetune).startswith('mae_pretrain_vit'):
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
    elif os.path.basename(args.finetune).startswith('swin_base_patch4'):
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
    else:
        raise ValueError(os.path.basename(args.finetune))

    if args.is_OOD_train:
        transform_train = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
    else:
        transform_train = transforms.Compose([
            RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    
    dataset_train = general_dataset_few_shot(args.source_data_path, args.source_dataset, args.target_data_path, args.target_dataset, train=True, 
                                             transform=transform_train, shot=args.few_shot_shot, seed=args.few_shot_seed,
                                            )
    dataset_val = general_dataset_few_shot(args.source_data_path, args.source_dataset,args.target_data_path, args.target_dataset, train=False, 
                                           transform=transform_val, shot=args.few_shot_shot, seed=args.few_shot_seed,
                                           )
    if 'oxford_flowers' in args.dataset:
        nb_classes = 102
    elif 'food-101' in args.dataset:
        nb_classes = 101
    elif 'fgvc_aircraft' in args.dataset:
        nb_classes = 100
    elif 'imagenet' in args.dataset:
        nb_classes = 1000
    
    return dataset_train, dataset_val, nb_classes
