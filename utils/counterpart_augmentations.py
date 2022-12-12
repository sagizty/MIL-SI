"""
counterparts online augmentations   Script  ver： Apr 14th 18:30
修改：
cutout返回的label也是long以使用cross entrophy
"""

# 博客连接：https://blog.csdn.net/cp1314971/article/details/106612060
import numpy as np
import torch
from scipy.special import perm


# 产生随机边界
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# augmentation SAMPLE
class Cutout(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        self.labels = torch.eye(self.class_num)[labels, :].to(self.device)  # 转为独热码
        self.inputs = inputs.clone().to(self.device)  # 存储inputs
        self.cutout_inputs = inputs.clone().to(self.device)  # 存储处理后的inputs
        self.act = act
        self.lam = []  # 存储lam

        for i in range(self.batch_size):

            if np.random.randint(0, 101) > 100 * self.p or (not self.act):
                # 根据概率判断是否进行增强
                self.lam.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)  # 计算裁剪部分外框的坐标

            self.cutout_inputs[i, :, bbx1:bbx2, bby1:bby2] = 0

            # 更新裁剪区域所占面积的比例
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[2] * inputs.size()[3]))
            self.lam.append(lam)

        self.long_label = self.labels.argmax(dim=1)

        # 注意，cutout返回的label也是long以使用cross entrophy
        return self.cutout_inputs, self.long_label, self.long_label


class CutMix(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size

        # 对数据增强概率进行矫正，矫正后的p为一个batch中有图片进行数据增强的概率
        self.p = shuffle_p * (perm(self.batch_size, self.batch_size)
                              / (perm(self.batch_size, self.batch_size) - 1))
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):

        self.labels = torch.eye(self.class_num)[labels, :].to(self.device)  # 转为独热码
        self.inputs = inputs.clone().to(self.device)  # 存储inputs
        self.cutmix_inputs = inputs.clone().to(self.device)  # 存储处理后的inputs
        self.lam = []
        self.act = act
        self.indices = torch.randperm(self.batch_size, device=self.device)  # 存储交换后图片的下标
        self.shuffled_inputs = inputs[self.indices].to(self.device)  # 存储交换顺序后的图片
        self.shuffled_labels = self.labels[self.indices].to(self.device)  # 存储交换顺序后的图片的标签

        for i in range(self.batch_size):

            if np.random.randint(0, 101) > 100 * self.p or (not self.act):
                self.lam.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)

            self.cutmix_inputs[i, :, bbx1:bbx2, bby1:bby2] = \
                self.shuffled_inputs[i, :, bbx1:bbx2, bby1:bby2]

            # 更新原图所占面积的比例
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[2] * inputs.size()[3]))
            self.lam.append(lam)
            self.labels[i] = self.labels[i] * lam + self.shuffled_labels[i] * (1 - lam)

        self.long_label = self.labels.argmax(dim=1)
        return self.cutmix_inputs, self.labels, self.long_label


class Mixup(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        self.p = shuffle_p * (perm(self.batch_size, self.batch_size)
                              / (perm(self.batch_size, self.batch_size) - 1))
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        self.labels = torch.eye(self.class_num)[labels, :].to(self.device)  # 转为独热码
        self.inputs = inputs.clone().to(self.device)  # 存储inputs
        self.mixup_inputs = inputs.clone().to(self.device)  # 存储处理后的inputs
        self.lam = []
        self.act = act
        self.indices = torch.randperm(self.batch_size, device=self.device)
        self.shuffled_inputs = inputs[self.indices].to(self.device)
        self.shuffled_labels = self.labels[self.indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not self.act):
                self.lam.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            self.lam.append(lam)
            self.mixup_inputs[i] = self.inputs[i] * lam + self.shuffled_inputs[i] * (1 - lam)
            self.labels[i] = self.labels[i] * lam + self.shuffled_labels[i] * (1 - lam)

        self.long_label = self.labels.argmax(dim=1)
        return self.mixup_inputs, self.labels, self.long_label


# ask func
def get_counterpart_augmentation(augmentation_name, p=1.0, class_num=2, batch_size=4, device='cpu'):
    """
    :param augmentation_name: 数据增强方式
    :param p: 进行增强的概率
    :param class_num: 类别数
    :param batch_size: batch size
    其中augmentation_name，class_num，batch_size必须提供
    """
    if augmentation_name == 'Cutout':
        Augmentation = Cutout(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)
        return Augmentation

    elif augmentation_name == 'CutMix':
        Augmentation = CutMix(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)
        return Augmentation

    elif augmentation_name == 'Mixup':
        Augmentation = Mixup(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)
        return Augmentation

    else:
        print('no valid counterparts augmentation selected')
        return None


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    label = torch.randn(4).long()

    Augmentation_1 = get_counterpart_augmentation('Cutout')
    inputs, labels = Augmentation_1(x, label)

    print(inputs, labels)
