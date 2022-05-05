"""
schedulers   Script  ver： May 1st 19:00

lr_scheduler from MAE code.
https://github.com/facebookresearch/mae

puzzle_patch_scheduler is used to arrange patch size for multi-scale learning

"""

import math
import random


def factor(num):  # 求因数
    factors = []
    for_times = int(math.sqrt(num))
    for i in range(for_times + 1)[1:]:
        if num % i == 0:
            factors.append(i)
            t = int(num / i)
            if not t == i:
                factors.append(t)
    return factors


def defactor(num_list, basic_num):  # 求倍数
    array = []
    for i in num_list:
        if i // basic_num * basic_num - i == 0:
            array.append(i)
    array.sort()  # accend
    return array


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup
    epoch，不一定是int，这个也支持float，从而通过中间epoch位置，来实现更精确的调节学习率：data_iter_step / len(data_loader) + epoch
    """
    # 计算目前epoch要的lr
    if epoch < args.warmup_epochs:  # 首先是warmup
        lr = args.lr * epoch / args.warmup_epochs  # lr从0线性上升到lr

    else:  # 对于之后的lr，采用余铉退火
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    # 更新optimizer里的lr
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class patch_scheduler:

    def __init__(self, total_epoches=200, warmup_epochs=20, edge_size=384, basic_patch=16, strategy=None):
        super().__init__()

        self.strategy = strategy

        self.total_epoches = total_epoches
        self.warmup_epochs = warmup_epochs

        self.patch_list = defactor(factor(edge_size), basic_patch)

        # No need for patch at all fig level
        if len(self.patch_list) > 1:
            self.patch_list = self.patch_list[:-1]

        if self.strategy == 'reverse':  # 先学习大的拼接，再学习小的拼接
            self.patch_list.sort(reverse=True)

        # self.loss_log ?
        # TODO 暂时不知道写啥strategy

    def __call__(self, epoch):
        if self.strategy is None:
            puzzle_patch_size = 32

        elif self.strategy == 'linear' or self.strategy == 'reverse':  # reverse 先学习大的拼接，再学习小的拼接
            if epoch < self.warmup_epochs:  # 首先是warmup
                puzzle_patch_size = 32  # 暂时采用固定的patch做warm up
            else:
                puzzle_patch_size = self.patch_list[min(int((epoch - self.warmup_epochs)
                                                            / (self.total_epoches - self.warmup_epochs)
                                                            * len(self.patch_list)), len(self.patch_list) - 1)]

        elif self.strategy == 'loop':
            # 每隔group_size个epoch，改变一次puzzle size，依次循环学习不同尺寸的拼接
            group_size = 3

            if epoch < self.warmup_epochs:
                puzzle_patch_size = 32  # in warm up epoches, fixed patch size
            else:
                group_idx = (epoch - self.warmup_epochs) % (len(self.patch_list) * group_size)
                puzzle_patch_size = self.patch_list[int(group_idx / group_size)]

        elif self.strategy == 'random':  # 随机学习不同尺寸的拼接
            puzzle_patch_size = random.choice(self.patch_list)

        else:  # TODO 暂时不知道写啥strategy, loss-drive?
            puzzle_patch_size = self.patch_list[-1]  # basic_patch

        return puzzle_patch_size


'''
scheduler = puzzle_patch_scheduler(strategy='reverse')
epoch = 182
puzzle_patch_size = scheduler(epoch)
print(puzzle_patch_size)
'''


class ratio_scheduler:
    def __init__(self, total_epoches=200, warmup_epochs=20, basic_ratio=0.25, strategy=None):
        super().__init__()
        self.strategy = strategy

        self.total_epoches = total_epoches
        self.warmup_epochs = warmup_epochs

        self.basic_ratio = basic_ratio

    def __call__(self, epoch):
        if self.strategy is None:
            fix_position_ratio = self.basic_ratio

        else:  # ratio逐步下降
            if epoch < self.warmup_epochs:  # 首先是warmup
                fix_position_ratio = self.basic_ratio  # 暂时采用固定的patch做warm up
            else:
                max_ratio = min(3 * self.basic_ratio, 0.9)  # upper-limit of 0.9
                min_ratio = max(self.basic_ratio * 0.5, 0.1)  # lower-limit of 0.1 ?

                fix_position_ratio = min(max(((self.total_epoches - self.warmup_epochs)
                                              - (epoch - self.warmup_epochs)) /
                                             (self.total_epoches - self.warmup_epochs)
                                             * max_ratio, min_ratio), max_ratio)

        return fix_position_ratio


'''
scheduler = puzzle_fix_position_ratio_scheduler(strategy='reverse')
epoch = 102
fix_position_ratio = scheduler(epoch)
print(fix_position_ratio)
'''
