"""
data_augmentation    Script  ver： Sep 1st 20:30

dataset structure: ImageNet
image folder dataset is used.
"""

from torchvision import transforms


def data_augmentation(data_augmentation_mode=0, edge_size=384):
    if data_augmentation_mode == 0:  # ROSE + MARS
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(700),  # center area for classification
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(700),
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }
        
    elif data_augmentation_mode == 1:  # Cervical
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }

    elif data_augmentation_mode == 2:  # warwick
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(360),  # center area for classification
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(360),
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }

    elif data_augmentation_mode == 3:  # for the squre input: just resize
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }
    else:
        print('no legal data augmentation is selected')
        return -1
    return data_transforms

