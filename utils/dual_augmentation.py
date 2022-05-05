"""
dual augmentation on both images and their masks   Script  ver： Apr 29th 14:00


"""
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from utils.tools import to_2tuple


class DualCompose:  # fit pytorch transform
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
            # NOTICE 转回图片 值总和还变成了cv2 numpy的1/255

        # Trans cv2 BGR image to PIL RGB image
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        b, g, r = cv2.split(mask)
        mask = cv2.merge([r, g, b])

        return Image.fromarray(np.uint8(image)), Image.fromarray(np.uint8(mask))


class DualImageTransform:
    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        # Trans cv2 BGR image to PIL RGB image
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        b, g, r = cv2.split(mask)
        mask = cv2.merge([r, g, b])
        return Image.fromarray(np.uint8(image)), Image.fromarray(np.uint8(mask))


class Dual_RandomHorizontalFlip:
    """
    Random horizontal flip.
    image shape: (height, width, channels)
    mask shape: (height, width)
    possibility: possibility for flip
    """

    def __init__(self, possibility=0.5):
        assert isinstance(possibility, (int, float))
        self.possibility = possibility

    def __call__(self, image, mask):
        if random.random() <= self.possibility:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        return image, mask


class Dual_RandomVerticalFlip:
    """
    Random vertical flip.
    image shape: (height, width, channels)
    mask shape: (height, width)
    possibility: possibility for flip
    """

    def __init__(self, possibility=0.5):
        assert isinstance(possibility, (int, float))
        self.possibility = possibility

    def __call__(self, image, mask):
        if random.random() <= self.possibility:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        return image, mask


class Dual_Rotate:
    """
    Random rotation.
    image shape: (height, width, channels)
    mask shape: (height, width)
    possibility: possibility for rotate
    range: range of rotation angles
    """

    def __init__(self, possibility=0.5, range=20):
        self.possibility = possibility
        self.range = range

    def __call__(self, image, mask):
        # 这里cv2读到的是反的，因此这里是height, width而不是width，height，图片input不是正方形时会有严重后果
        height, width = image.shape[:2]

        if random.random() <= self.possibility:
            angle = np.random.randint(0, self.range)

            center = (width // 2, height // 2)
            # 得到旋转矩阵，第一个参数为旋转中心，第二个参数为旋转角度，第三个参数为旋转之前原图像缩放比例
            M = cv2.getRotationMatrix2D(center, -angle, 1)
            # 进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
            image = cv2.warpAffine(image, M, (width, height))
            mask = cv2.warpAffine(mask.astype(np.uint8), M, (width, height))

        return image.astype(np.uint8), mask.astype(np.int)


def Four_step_dual_augmentation(data_augmentation_mode=0, edge_size=384):
    """
    Get data augmentation methods

    Dual_transform : Transform CV2 images and their mask by Rotate, RandomHorizontalFlip, etc.
    DualImage : Transform CV2 images and their mask to PIL images
    train_domain_transform : transforms.ColorJitter on PIL images
    transform: PIL crop, resize and to Tensor

    USAGE:

    IN Train:
    image, mask = self.Dual_transform(image, mask)
    # image color jitter shifting
    image = self.train_domain_transform(image)
    # crop + resize
    image = self.transform(image)

    IN Val $ Test:

    # 0/255 mask -> binary mask
    image, mask = self.DualImage(image, mask)
    # crop + resize
    image = self.transform(image)
    """

    edge_size = to_2tuple(edge_size)

    if data_augmentation_mode == 0:  # ROSE + MARS
        # apply the on-time synchornized transform on image and mask togather
        Dual_transform = DualCompose([
            Dual_Rotate(possibility=0.8, range=180),
            Dual_RandomHorizontalFlip(),
            Dual_RandomVerticalFlip(),
        ])
        # val & test use DualImage to convert PIL Image
        DualImage = DualImageTransform()

        # ColorJitter for image only
        train_domain_transform = transforms.Compose([
            # HSL shift operation
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
        ])

        # lastly, the synchornized separate transform
        transform = transforms.Compose([
            transforms.CenterCrop(700),  # center area for classification
            transforms.Resize(edge_size),
            transforms.ToTensor(),  # hwc -> chw tensor
        ])

    elif data_augmentation_mode == 1:  # Cervical
        # apply the on-time synchornized transform on image and mask togather
        Dual_transform = DualCompose([
            Dual_Rotate(possibility=0.8, range=180),
            Dual_RandomHorizontalFlip(),
            Dual_RandomVerticalFlip(),
        ])
        # val & test use DualImage to convert PIL Image
        DualImage = DualImageTransform()

        # ColorJitter for image only
        train_domain_transform = transforms.Compose([
            # HSL shift operation
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
        ])

        # lastly, the synchornized separate transform
        transform = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.ToTensor(),  # hwc -> chw tensor
        ])

    elif data_augmentation_mode == 2:  # warwick
        # apply the on-time synchornized transform on image and mask togather
        Dual_transform = DualCompose([
            Dual_Rotate(possibility=0.8, range=180),
            Dual_RandomHorizontalFlip(),
            Dual_RandomVerticalFlip(),
        ])
        # val & test use DualImage to convert PIL Image
        DualImage = DualImageTransform()

        # ColorJitter for image only
        train_domain_transform = transforms.Compose([
            # HSL shift operation
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
        ])

        # lastly, the synchornized separate transform
        transform = transforms.Compose([
            transforms.CenterCrop(360),  # center area for classification
            transforms.Resize(edge_size),
            transforms.ToTensor(),  # hwc -> chw tensor
        ])

    elif data_augmentation_mode == 3:  # TODO 对于方形输入
        # apply the on-time synchornized transform on image and mask togather
        Dual_transform = DualCompose([
            # Dual_Rotate(possibility=0.8, range=180),
            Dual_RandomHorizontalFlip(),
            Dual_RandomVerticalFlip(),
        ])
        # val & test use DualImage to convert PIL Image
        DualImage = DualImageTransform()

        # ColorJitter for image only
        train_domain_transform = transforms.Compose([
            # HSL shift operation
            transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
        ])

        # lastly, the synchornized separate transform
        transform = transforms.Compose([
            transforms.CenterCrop(360),  # center area for classification
            transforms.Resize(edge_size),
            transforms.ToTensor(),  # hwc -> chw tensor
        ])

    else:
        print('no legal data augmentation is selected')
        return -1

    return Dual_transform, DualImage, train_domain_transform, transform
