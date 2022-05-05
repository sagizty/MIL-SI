"""
Training   Script  ver： Apr 24th 22:00

dataset structure: ImageNet
image folder dataset is used.

update
SoftlabelCrossEntropy loss for soft-label based augmentations
fixme 好像说reduction='sum' 有问题？
"""

from __future__ import print_function, division

import argparse
import json

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch import Tensor

import torchvision
from torchsummary import summary
import time

import copy
from tensorboardX import SummaryWriter

from utils.visual_usage import *
from utils.data_augmentation import *
from utils.counterpart_augmentations import *
from utils.tools import setup_seed, del_file, FixStateDict
from MIL.getmodel import get_model


# define SoftlabelCrossEntropy loss for soft-label based augmentations
def SoftCrossEntropy(input, target, reduction='sum'):  # reduction='sum' fixme 好像说有问题？查一下warning
    log_likelihood = -F.log_softmax(input, dim=1)
    batch = input.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class SoftlabelCrossEntropy(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'sum') -> None:
        super(SoftlabelCrossEntropy, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return SoftCrossEntropy(input, target, reduction=self.reduction)


# Training Script
def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # determin which epoch have the best model

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, Augmentation=None, edge_size=384,
                model_idx=None, num_epochs=25, intake_epochs=0, check_minibatch=100, scheduler=None, device=None,
                draw_path='../imagingresults', enable_attention_check=False, enable_visualize_check=False,
                enable_sam=False, writer=None):
    """
    Training iteration

    :param model: model object
    :param dataloaders: 2 dataloader(train and val) dict
    :param criterion: loss func obj
    :param optimizer: optimizer obj

    :param class_names: The name of classes for priting
    :param dataset_sizes: size of datasets

    :param Augmentation: Online augmentation

    :param edge_size: image size for the input image

    :param model_idx: model idx for the getting pre-setted model

    :param num_epochs: total training epochs
    :param intake_epochs: number of skip over epochs when choosing the best model
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.
    :param scheduler: scheduler is an LR scheduler object from torch.optim.lr_scheduler.

    :param device: cpu/gpu object
    :param draw_path: path folder for output pic

    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics
    :param enable_sam: use SAM training strategy

    :param writer: attach the records to the tensorboard backend
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # for saving the best model state dict
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy
    # initial an empty dict
    json_log = {}

    # initial best performance
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    best_epoch_idx = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # record json log, initially empty
        json_log[str(epoch + 1)] = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # alternatively train/val

            index = 0
            model_time = time.time()

            # initiate the empty log dict
            log_dict = {}
            for cls_idx in range(len(class_names)):
                # only float type is allowed in json, set to float inside
                log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # criterias, initially empty
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # use different dataloder in different phase

                inputs = inputs.to(device)  # print('inputs[0]',type(inputs[0]))
                # NOTICE in CLS task the labels' type is long tensor([B])，not one-hot ([B,CLS])
                labels = labels.to(device)

                # Online Augmentation of Cutmix/Cutout/Mixup on device
                if Augmentation is not None:
                    if phase == 'train':
                        inputs, labels, GT_long_labels = Augmentation(inputs, labels)
                    else:  # Val
                        inputs, labels, GT_long_labels = Augmentation(inputs, labels, act=False)
                else:
                    GT_long_labels = labels  # store ori_label on CPU

                # zero the parameter gradients
                if not enable_sam:
                    optimizer.zero_grad()

                # forward
                # track grad if only in train!
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)  # one-hot outputs: [B,CLS]
                    _, preds = torch.max(outputs, 1)  # idx outputs: [B] each is a idx
                    loss = criterion(outputs, labels)  # cross entrphy of one-hot outputs: [B,CLS] and idx label [B]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if enable_sam:
                            loss.backward()
                            # first forward-backward pass
                            optimizer.first_step(zero_grad=True)

                            # second forward-backward pass
                            loss2 = criterion(model(inputs), labels)  # SAM need another model(inputs)
                            loss2.backward()  # make sure to do a full forward pass when using SAM
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()

                # log criterias: update
                log_running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.cpu() == GT_long_labels.cpu().data)

                # Compute precision and recall for each class.
                for cls_idx in range(len(class_names)):
                    tp = np.dot((GT_long_labels.cpu().data == cls_idx).numpy().astype(int),
                                (preds == cls_idx).cpu().numpy().astype(int))
                    tn = np.dot((GT_long_labels.cpu().data != cls_idx).numpy().astype(int),
                                (preds != cls_idx).cpu().numpy().astype(int))

                    fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

                    fn = np.sum((GT_long_labels.cpu().data == cls_idx).numpy()) - tp

                    # log_dict[cls_idx] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0} set to float inside
                    log_dict[class_names[cls_idx]]['tp'] += tp
                    log_dict[class_names[cls_idx]]['tn'] += tn
                    log_dict[class_names[cls_idx]]['fp'] += fp
                    log_dict[class_names[cls_idx]]['fn'] += fn

                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' minibatch loss',
                                      float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' minibatch ACC',
                                      float(torch.sum(preds.cpu() == GT_long_labels.cpu().data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # at the checking time now
                if index % check_minibatch == check_minibatch - 1:
                    model_time = time.time() - model_time

                    check_index = index // check_minibatch + 1

                    epoch_idx = epoch + 1
                    print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                          check_index, '     time used:', model_time)

                    print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

                    if enable_visualize_check:
                        visualize_check(inputs, GT_long_labels, model, class_names, num_images=3,
                                        pic_name='Visual_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                        draw_path=draw_path, writer=writer)

                    if enable_attention_check:
                        try:
                            check_SAA(inputs, GT_long_labels, model, model_idx, edge_size, class_names, num_images=1,
                                      pic_name='GradCAM_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                      draw_path=draw_path, writer=writer)
                        except:
                            print('model:', model_idx, ' with edge_size', edge_size, 'is not supported yet')
                    else:
                        pass

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

            if phase == 'train':
                if scheduler is not None:  # lr scheduler: update
                    scheduler.step()

            # log criterias: print
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            print('\nEpoch: {}  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))

            # attach the records to the tensorboard backend
            if writer is not None:
                # ...log the running loss
                writer.add_scalar(phase + ' loss',
                                  float(epoch_loss),
                                  epoch + 1)
                writer.add_scalar(phase + ' ACC',
                                  float(epoch_acc),
                                  epoch + 1)

            # calculating the confusion matrix
            for cls_idx in range(len(class_names)):
                tp = log_dict[class_names[cls_idx]]['tp']
                tn = log_dict[class_names[cls_idx]]['tn']
                fp = log_dict[class_names[cls_idx]]['fp']
                fn = log_dict[class_names[cls_idx]]['fn']
                tp_plus_fp = tp + fp
                tp_plus_fn = tp + fn
                fp_plus_tn = fp + tn
                fn_plus_tn = fn + tn

                # precision
                if tp_plus_fp == 0:
                    precision = 0
                else:
                    precision = float(tp) / tp_plus_fp * 100
                # recall
                if tp_plus_fn == 0:
                    recall = 0
                else:
                    recall = float(tp) / tp_plus_fn * 100

                # TPR (sensitivity)
                TPR = recall

                # TNR (specificity)
                # FPR
                if fp_plus_tn == 0:
                    TNR = 0
                    FPR = 0
                else:
                    TNR = tn / fp_plus_tn * 100
                    FPR = fp / fp_plus_tn * 100

                # NPV
                if fn_plus_tn == 0:
                    NPV = 0
                else:
                    NPV = tn / fn_plus_tn * 100

                print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
                print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
                print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
                print('{} TP: {}'.format(class_names[cls_idx], tp))
                print('{} TN: {}'.format(class_names[cls_idx], tn))
                print('{} FP: {}'.format(class_names[cls_idx], fp))
                print('{} FN: {}'.format(class_names[cls_idx], fn))
                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' precision',
                                      precision,
                                      epoch + 1)
                    writer.add_scalar(phase + '   ' + class_names[cls_idx] + ' recall',
                                      recall,
                                      epoch + 1)

            # json log: update
            json_log[str(epoch + 1)][phase] = log_dict

            if phase == 'val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # not useful actually

            # deep copy the model
            if phase == 'val' and better_performance(temp_acc, temp_vac, best_acc, best_vac) and epoch >= intake_epochs:
                # what is better? we now use the wildly used method only
                best_epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())
                best_log_dic = log_dict

            print('\n')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch idx: ', best_epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))
    for cls_idx in range(len(class_names)):
        tp = best_log_dic[class_names[cls_idx]]['tp']
        tn = best_log_dic[class_names[cls_idx]]['tn']
        fp = best_log_dic[class_names[cls_idx]]['fp']
        fn = best_log_dic[class_names[cls_idx]]['fn']
        tp_plus_fp = tp + fp
        tp_plus_fn = tp + fn
        fp_plus_tn = fp + tn
        fn_plus_tn = fn + tn

        # precision
        if tp_plus_fp == 0:
            precision = 0
        else:
            precision = float(tp) / tp_plus_fp * 100
        # recall
        if tp_plus_fn == 0:
            recall = 0
        else:
            recall = float(tp) / tp_plus_fn * 100

        # TPR (sensitivity)
        TPR = recall

        # TNR (specificity)
        # FPR
        if fp_plus_tn == 0:
            TNR = 0
            FPR = 0
        else:
            TNR = tn / fp_plus_tn * 100
            FPR = fp / fp_plus_tn * 100

        # NPV
        if fn_plus_tn == 0:
            NPV = 0
        else:
            NPV = tn / fn_plus_tn * 100

        print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
        print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
        print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # load best model weights as final model training result
    model.load_state_dict(best_model_wts)
    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)
    return model


def main(args):
    if args.paint:
        # use Agg kernal, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    enable_notify = args.enable_notify  # True
    enable_tensorboard = args.enable_tensorboard  # True
    enable_attention_check = args.enable_attention_check  # False   'CAM' 'SAA'
    enable_visualize_check = args.enable_visualize_check  # False

    enable_sam = args.enable_sam  # False

    data_augmentation_mode = args.data_augmentation_mode  # 0

    linearprobing = args.linearprobing  # False

    Pre_Trained_model_path = args.Pre_Trained_model_path  # None

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    # model info
    model_idx = args.model_idx  # the model we are going to use. by the format of Model_size_other_info
    # structural parameter
    drop_rate = args.drop_rate
    attn_drop_rate = args.attn_drop_rate
    drop_path_rate = args.drop_path_rate
    use_cls_token = False if args.cls_token_off else True
    use_pos_embedding = False if args.pos_embedding_off else True
    use_att_module = None if args.att_module == 'None' else args.att_module

    # pretrained_backbone
    pretrained_backbone = False if args.backbone_PT_off else True

    # classification required number of your dataset
    num_classes = args.num_classes  # default 0 for auto-fit
    # image size for the input image
    edge_size = args.edge_size  # 224 384 1000

    # batch info
    batch_size = args.batch_size  # 8
    num_workers = args.num_workers  # main training num_workers 4

    num_epochs = args.num_epochs  # 50
    intake_epochs = args.intake_epochs  # 0
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 400 // batch_size

    lr = args.lr  # 0.000007
    lrf = args.lrf  # 0.0

    opt_name = args.opt_name  # 'Adam'

    # PATH info
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot

    draw_path = os.path.join(draw_root, 'CLS_' + model_idx)  # PC is for the pancreatic cancer
    save_model_path = os.path.join(model_path, 'CLS_' + model_idx + '.pth')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(draw_path)  # clear the output folder, NOTICE this may be DANGEROUS
    else:
        os.makedirs(draw_path)

    # Train Augmentation
    augmentation_name = args.augmentation_name  # None

    # Data Augmentation
    data_transforms = data_augmentation(data_augmentation_mode, edge_size=edge_size)

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(dataroot, x), data_transforms[x]) for x in
                ['train', 'val']}  # 2 dataset obj is prepared here and combine together
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}  # size of each dataset

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers, drop_last=True),  # colab suggest 2
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers // 4 + 1, drop_last=True)
                   }

    class_names = [d.name for d in os.scandir(os.path.join(dataroot, 'train')) if d.is_dir()]
    class_names.sort()
    if num_classes == 0:
        print("class_names:", class_names)
        num_classes = len(class_names)
    else:
        if len(class_names) == num_classes:
            print("class_names:", class_names)
        else:
            print('classfication number of the model mismatch the dataset requirement of:', len(class_names))
            return -1

    # use notifyemail to send the record to somewhere
    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='xxxxxxx', mail_pass='xxxxxxx',
                       default_reciving_list=['xxxxxxx'],  # fixme change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('update to the tensorboard')
        else:
            notify.add_text('not update to the tensorboard')

        notify.add_text('  ')

        notify.add_text('model idx ' + str(model_idx))
        notify.add_text('  ')

        notify.add_text('GPU idx: ' + str(gpu_idx))
        notify.add_text('  ')

        notify.add_text('cls number ' + str(num_classes))
        notify.add_text('class_names =' + str(class_names))
        notify.add_text('edge size ' + str(edge_size))
        notify.add_text('  ')
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('num_epochs ' + str(num_epochs))
        notify.add_text('lr ' + str(lr))
        notify.add_text('opt_name ' + str(opt_name))
        notify.add_text('  ')
        notify.add_text('enable_sam ' + str(enable_sam))
        notify.add_text('augmentation_name ' + str(augmentation_name))
        notify.add_text('data_augmentation_mode ' + str(data_augmentation_mode))

        notify.send_log()

    print("*********************************{}*************************************".format('setting'))
    print(args)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None
    # if u run locally
    # nohup tensorboard --logdir=/home/MSHT/runs --host=0.0.0.0 --port=7777 &
    # tensorboard --logdir=/home/ZTY/runs --host=0.0.0.0 --port=7777

    if gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = gpu_idx
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # setting k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
            gpu_use = gpu_idx
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'

    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model
    # get model: randomly initiate model, except the backbone CNN(when pretrained_backbone is True)
    model = get_model(num_classes, edge_size, model_idx, drop_rate, attn_drop_rate, drop_path_rate,
                      pretrained_backbone, use_cls_token, use_pos_embedding, use_att_module)

    # Manually get the model pretrained on the Imagenet1000
    if Pre_Trained_model_path is not None:
        if os.path.exists(Pre_Trained_model_path):
            state_dict = FixStateDict(torch.load(Pre_Trained_model_path), remove_key_head='head')
            model.load_state_dict(state_dict, False)
            print('pretrain model loaded')
        else:
            print('Pre_Trained_model_path:' + Pre_Trained_model_path, ' is NOT avaliable!!!!\n')
            raise  # print('we ignore this with a new start up')

    if linearprobing:
        # Only tuning the last FC layer for CLS task
        module_all = 0
        for child in model.children():  # find all nn.modules
            module_all += 1

        for param in model.parameters():  # freeze all parameters
            param.requires_grad = False

        for module_idx, child in enumerate(model.children()):
            if module_idx == module_all:  # Unfreeze the parameters of the last FC layer
                for param in child.parameters():
                    param.requires_grad = True
        

    print('GPU:', gpu_use)

    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    try:
        summary(model, input_size=(3, edge_size, edge_size))  # should be after .to(device)
    except:
        pass

    print("model :", model_idx)

    # Augmentation
    Augmentation = get_counterpart_augmentation(augmentation_name, p=0.6667, class_num=num_classes,
                                                batch_size=batch_size, device=device)

    # Default cross entrphy of one-hot outputs: [B,CLS] and idx label [B] long tensor
    # augmentation loss is SoftlabelCrossEntropy
    criterion = SoftlabelCrossEntropy() \
        if Augmentation is not None and augmentation_name != 'Cutout' else nn.CrossEntropyLoss()

    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 15 0.1  default SGD StepLR scheduler
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = None
    else:
        print('no optimizer')
        raise

    if enable_sam:
        from utils.sam import SAM

        if opt_name == 'SGD':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.8)
            scheduler = None
        elif opt_name == 'Adam':
            base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0.01)
        else:
            print('no optimizer')
            raise

    if lrf > 0:  # use cosine learning rate schedule
        import math
        # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    model_ft = train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, Augmentation,
                           edge_size=edge_size, model_idx=model_idx, num_epochs=num_epochs,
                           intake_epochs=intake_epochs, check_minibatch=check_minibatch,
                           scheduler=scheduler, device=device, draw_path=draw_path,
                           enable_attention_check=enable_attention_check,
                           enable_visualize_check=enable_visualize_check, enable_sam=enable_sam, writer=writer)

    # save model if its a multi-GPU model, save as a single GPU one too
    if gpu_use == -1:
        torch.save(model_ft.module.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)

    else:
        torch.save(model_ft.state_dict(), save_model_path)
        print('model trained by GPU (idx:' + str(gpu_use) + ') has been saved at ', save_model_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='Hybrid2_384_401_testsample', type=str, help='Model Name or index')
    # drop_rate, attn_drop_rate, drop_path_rate
    parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate , default 0.0')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='dropout rate Aftter Attention, default 0.0')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='drop path for stochastic depth, default 0.0')

    # Abalation Studies
    parser.add_argument('--cls_token_off', action='store_true', help='use cls_token in model structure')
    parser.add_argument('--pos_embedding_off', action='store_true', help='use pos_embedding in model structure')
    # 'SimAM', 'CBAM', 'SE' 'None'
    parser.add_argument('--att_module', default='SimAM', type=str, help='use which att_module in model structure')

    # backbone_PT_off  by default is false, in default setting the backbone weight is required
    parser.add_argument('--backbone_PT_off', action='store_true', help='use a freash backbone weight in training')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default='/data/MIL_Experiment/dataset/ROSE_CLS',
                        help='path to dataset')
    parser.add_argument('--model_path', default='/home/pancreatic-cancer-project/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default='/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Training status parameters
    # SAM
    parser.add_argument('--enable_sam', action='store_true', help='use SAM strategy in training')
    # Online augmentation_name
    parser.add_argument('--augmentation_name', default=None, type=str, help='Online augmentation name')
    # Dataset specific augmentations in dataloader
    parser.add_argument('--data_augmentation_mode', default=0, type=int, help='data_augmentation_mode')

    # Tuning setting
    # linearprobing
    parser.add_argument('--linearprobing', action='store_true', help='use linearprobing tuning')

    # Finetuning a Pretrained model at PATH
    # '/home/MIL_Experiment/saved_models/Hybrid2_384_PreTrain_000.pth'
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=0, type=int, help='classification number, default 0 for auto-fit')
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    parser.add_argument('--num_workers', default=2, type=int, help='use CPU num_workers , default 2 for colab')

    # Training seting parameters
    parser.add_argument('--batch_size', default=8, type=int, help='Training batch_size default 8')

    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    parser.add_argument('--num_epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--intake_epochs', default=0, type=int, help='only save model at epochs after intake_epochs')
    parser.add_argument('--lr', default=0.00001, type=float, help='learing rate')
    parser.add_argument('--lrf', type=float, default=0.0,
                        help='learing rate decay rate, default 0(not enabled), suggest 0.1 and lr=0.00005')
    parser.add_argument('--opt_name', default='Adam', type=str, help='optimizer name Adam or SGD')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
