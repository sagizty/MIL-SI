# 画ACC-Loss
from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器
import matplotlib.pyplot as plt
import os
import matplotlib


def find_all_files_startwith(root, suffix=None):
    """
    返回特定前缀的所有文件路径列表
    """
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.startswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def ACC_loss(PATH, out_file_path):
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)

    runs_all = find_all_files_startwith(PATH, suffix='events')
    print(runs_all)

    for runs_path in runs_all:
        model_idx = os.path.split(os.path.split(runs_path)[0])[1]

        ea = event_accumulator.EventAccumulator(runs_path)  # 初始化EventAccumulator对象
        ea.Reload()  # 这一步是必须的，将事件的内容都导进去
        # print(ea.scalars.Keys())  # 检查保存了哪些记录scalars

        train_ACC = ea.scalars.Items("train_ACC")
        train_loss = ea.scalars.Items("train_loss")  # 读取train_loss
        '''
        print([(i.step, i.value) for i in train_ACC])
        for i, j in zip(train_ACC, train_loss):
            print((i.value, j.value))
        '''
        ax1.plot([i.value for i in train_loss], [i.value for i in train_ACC], label=model_idx)

    plt.legend(loc='lower right')
    ax1.set_xlabel("Loss")
    ax1.set_ylabel("Acc")
    plt.show()
    plt.savefig(out_file_path, dpi=1000)


if __name__ == '__main__':
    matplotlib.use('Agg')
    PATH = './MIL-SI/Archive/log/abalation'
    out_file_path = './patch_size_abalation_loss-acc.jpg'
    ACC_loss(PATH, out_file_path)
