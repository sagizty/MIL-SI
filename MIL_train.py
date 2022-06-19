"""
MIL Training   Script  ver： Jun 5th 21:00
"""
from __future__ import print_function, division

import argparse
import copy
import json
import time

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchsummary import summary

from MIL.MIL_model import *
from MIL.MIL_structure import *
from utils.tools import setup_seed, del_file, FixStateDict
from utils.visual_usage import *


# Training Script
def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # determin which epoch have the best model

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False


def train(model, shuffle_patch_distributer, fixed_patch_distributer, dataloaders, MIL_criterion, CLS_criterion,
          optimizer, class_names, dataset_sizes, edge_size=384, CLS_MIL=True, model_idx=None, num_epochs=25,
          intake_epochs=0, check_minibatch=100, head_balance=(1., 1., 1.), scheduler=None, device=None,
          draw_path='../runs', enable_attention_check=False, enable_visualize_check=False, enable_sam=False,
          writer=None):

    if shuffle_patch_distributer is None:
        shuffle_MIL = False
    else:
        shuffle_MIL = True

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
                # only float type is allowed in json
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
            for patch_image, patch_labels, labels in dataloaders[phase]:  # use different dataloder in different phase

                # get datas
                patch_image = patch_image.to(device)
                patch_labels = patch_labels.to(device)
                labels = labels.to(device)

                # get bag data from the distributers
                if shuffle_MIL:
                    MIL_bag_image, MIL_bag_labels = shuffle_patch_distributer(patch_image, patch_labels)

                # CLS data and its MIL data (no shuffle)
                CLS_bag_image, CLS_bag_labels, labels = fixed_patch_distributer(patch_image, patch_labels, labels)

                # zero the parameter gradients
                if not enable_sam:
                    optimizer.zero_grad()

                # MIL-SI forward with 2-step(shuffle step + non-shuffle step)
                # track grad if only in train!
                with torch.set_grad_enabled(phase == 'train'):

                    # non-shuffle step: train both CLS_head and MIL_head with non-shuffled patches
                    # CLS_head for CLS task
                    bag_labels, outputs = model(CLS_bag_image, True)
                    _, preds = torch.max(outputs, 1)
                    CLS_loss = CLS_criterion(outputs, labels)

                    if CLS_MIL:  # MIL_head for soft_label on the non-shuffled patches
                        CLS_MIL_loss = MIL_criterion(bag_labels, CLS_bag_labels)

                    # shuffle step: train MIL_head with shuffled patches
                    if shuffle_MIL:
                        bag_labels = model(MIL_bag_image)
                        MIL_loss = MIL_criterion(bag_labels, MIL_bag_labels)

                    # compose loss for iteration
                    # head_balance = (CLS_weight, CLS_MIL_weight, MIL_weight)
                    if shuffle_MIL and CLS_MIL:
                        loss = head_balance[0] * CLS_loss \
                               + head_balance[1] * CLS_MIL_loss \
                               + head_balance[2] * MIL_loss  # todo 或许设置成对抗的loss组合方式？

                    elif not shuffle_MIL and CLS_MIL:
                        loss = head_balance[0] * CLS_loss + head_balance[1] * CLS_MIL_loss

                    elif shuffle_MIL and not CLS_MIL:
                        loss = head_balance[0] * CLS_loss \
                               + head_balance[2] * MIL_loss  # todo 或许设置成对抗的loss组合方式？
                    else:
                        loss = CLS_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if enable_sam:
                            loss.backward()
                            # first forward-backward pass
                            optimizer.first_step(zero_grad=True)

                            # second forward-backward pass
                            loss2 = CLS_criterion(model(patch_image), labels)  # SAM need another model(patch_image)
                            loss2.backward()  # make sure to do a full forward pass when using SAM
                            optimizer.second_step(zero_grad=True)
                        else:
                            loss.backward()
                            optimizer.step()

                # log criterias: update
                log_running_loss += loss.item()
                running_loss += loss.item() * patch_image.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Compute precision and recall for each class.
                for cls_idx in range(len(class_names)):
                    tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                                (preds == cls_idx).cpu().numpy().astype(int))
                    tn = np.dot((labels.cpu().data != cls_idx).numpy().astype(int),
                                (preds != cls_idx).cpu().numpy().astype(int))

                    fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

                    fn = np.sum((labels.cpu().data == cls_idx).numpy()) - tp

                    # log_dict[cls_idx] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}  # notice is float inside
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
                                      float(torch.sum(preds == labels.data) / patch_image.size(0)),
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
                        visualize_check(CLS_bag_image, labels, model, class_names, num_images=3,
                                        pic_name='Visual_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                        draw_path=draw_path, writer=writer)

                    if enable_attention_check:
                        try:
                            check_SAA(CLS_bag_image, labels, model, model_idx, edge_size, class_names,
                                      model_type='MIL', num_images=1,
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
                # TODO what is better? we now use the wildly used method only
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

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    enable_notify = args.enable_notify  # True
    enable_tensorboard = args.enable_tensorboard  # True
    enable_attention_check = args.enable_attention_check  # fixme when MIL head is used, CAM is wrong
    enable_visualize_check = args.enable_visualize_check  # False

    enable_sam = args.enable_sam  # False

    model_idx = args.model_idx
    # pretrained_backbone
    pretrained_backbone = False if args.backbone_PT_off else True

    # shuffle MIL approach
    shuffle_MIL = False if args.shuffle_MIL_off else True  # True
    # CLS step MIL head, set False to use only CLS head in the no-shuffle step, by default True
    CLS_MIL = False if args.CLS_MIL_off else True  # True

    # image size and patch_size for the input MIL image
    edge_size = args.edge_size  # 224 384
    patch_size = args.patch_size

    batch_size = args.batch_size  # GPU memory cost: colab 4  gpu server 8
    num_workers = args.num_workers  # cpu server 0    colab suggest 2    gpu server 8 ?

    lr = args.lr
    lrf = args.lrf  # 0.0

    head_balance = (1., 1. * args.CLS_MIL_head_weight, 1. * args.MIL_head_weight)

    opt_name = args.opt_name  # 'Adam'

    # PATH info
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot

    Pre_Trained_model_path = args.Pre_Trained_model_path  # None

    data_augmentation_mode = args.data_augmentation_mode  # 0

    num_epochs = args.num_epochs  # 50
    intake_epochs = args.intake_epochs  # 0
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 400 // batch_size

    draw_path = os.path.join(draw_root, 'MIL_' + model_idx)  # PC is for the plant cls, MIL for MIL task
    save_model_path = os.path.join(model_path, 'MIL_' + model_idx + '.pth')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if os.path.exists(draw_path):
        del_file(draw_path)  # clear the output folder, NOTICE this may be DANGEROUS
    else:
        os.makedirs(draw_path)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None

    # device
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
    print('GPU:', gpu_use)

    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 dataset obj is prepared here and combine together
    datasets = {x: MILDataset(os.path.join(dataroot, x), mode=x, data_augmentation_mode=data_augmentation_mode,
                              suffix='.jpg', edge_size=edge_size, patch_size=patch_size) for x in ['train', 'val']}

    class_names = datasets['train'].class_names

    dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers),  # colab suggest 2
                   'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers)
                   }

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}  # size of each dataset

    if enable_notify:  # use notifyemail to send the record to somewhere
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='xxxx@163.com', mail_pass='xxxx',
                       default_reciving_list=['xxxx@163.com'],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('update to the tensorboard')
        else:
            notify.add_text('not update to the tensorboard')

        notify.add_text('  ')

        notify.add_text('model idx ' + str(model_idx))
        notify.add_text('  ')

        notify.add_text('GPU use: ' + str(gpu_use))
        notify.add_text('  ')

        notify.add_text('cls number ' + str(len(class_names)))
        notify.add_text('edge size ' + str(edge_size))
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('MIL_patch_size ' + str(patch_size))

        notify.add_text('num_epochs ' + str(num_epochs))
        notify.add_text('lr ' + str(lr))
        notify.add_text('opt_name ' + str(opt_name))
        notify.add_text('enable_sam ' + str(enable_sam))
        notify.send_log()

    print("*********************************{}*************************************".format('setting'))
    print(args)

    # 2-step data distributers
    if shuffle_MIL:
        # shuffle step data distributer
        shuffle_patch_distributer = shuffle_distributer(edge_size, patch_size, device=device)
    else:
        # set the shuffle step data distributer to None
        shuffle_patch_distributer = None
    # non-shuffle step data distributer
    fixed_patch_distributer = non_shuffle_distributer(edge_size, patch_size, device=device)

    # model
    model = build_MIL_model(model_idx, edge_size, pretrained_backbone, num_classes=len(class_names))

    # get Pre_Trained model if required
    if Pre_Trained_model_path is not None:
        if os.path.exists(Pre_Trained_model_path):
            state_dict = FixStateDict(torch.load(Pre_Trained_model_path), remove_key_head='head')
            model.load_state_dict(state_dict, False)
            print('pretrain model loaded')
        else:
            print('Pre_Trained_model_path:' + Pre_Trained_model_path, ' is NOT avaliable!!!!\n')
            raise  # print('we ignore this with a new start up')

    # put on multi-gpu
    if gpu_use == -1:
        model = nn.DataParallel(model)
    model.to(device)

    try:
        summary(model, input_size=(3, edge_size, edge_size))  # should be after .to(device)
    except:
        pass
    print("model :", model_idx)

    # Training setting
    MIL_criterion = nn.L1Loss(size_average=None, reduce=None)  # todo find better loss, maybe smoothl1loss？
    CLS_criterion = nn.CrossEntropyLoss()

    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 15 0.1  default SGD StepLR scheduler
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = None
    else:
        print('no optimizer')
        raise

    # SAM
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

    # Train
    model_ft = train(model, shuffle_patch_distributer, fixed_patch_distributer, dataloaders, MIL_criterion, CLS_criterion,
                     optimizer, class_names=class_names, dataset_sizes=dataset_sizes, edge_size=edge_size,
                     CLS_MIL=CLS_MIL, model_idx=model_idx, num_epochs=num_epochs, intake_epochs=intake_epochs,
                     check_minibatch=check_minibatch, head_balance=head_balance, scheduler=scheduler, device=device,
                     draw_path=draw_path, enable_attention_check=enable_attention_check,
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
    parser.add_argument('--model_idx', default='ViT_384_401_PT_lf05_b4_ROSE_MIL', type=str, help='Model Name or index')

    # backbone_PT_off  by default is false, in default setting the backbone weight is required
    parser.add_argument('--backbone_PT_off', action='store_true', help='use a freash backbone weight in training')

    # shuffle_MIL, set --shuffle_MIL_off to use only the no-shuffle step to train 2-head, by default use both steps
    parser.add_argument('--shuffle_MIL_off', action='store_true', help='disable shuffle MIL training')
    # no-shuffle step (CLS step) MIL head, set --CLS_MIL_off to use only CLS head in the no-shuffle step
    parser.add_argument('--CLS_MIL_off', action='store_true', help='disable CLS_MIL training')

    # Trained models
    # '/home/MIL_Experiment/saved_models/Hybrid2_384_PreTrain_000.pth'
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default='/data/MIL_Experiment/dataset/ROSE_MIL',
                        help='path to dataset')
    parser.add_argument('--model_path', default='/home/MIL_Experiment/saved_models',
                        help='path to save model state-dict')
    parser.add_argument('--draw_root', default='/home/MIL_Experiment/runs',
                        help='path to draw and save tensorboard output')

    # Data flow parameters
    parser.add_argument('--data_augmentation_mode', default=0, type=int, help='data_augmentation_mode')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Training status parameters
    parser.add_argument('--enable_sam', action='store_true', help='use SAM strategy in training')

    # Dataset based parameters
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000
    parser.add_argument('--num_workers', default=2, type=int, help='use CPU num_workers , default 2 for colab')

    # Training seting parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Training batch_size default 8')

    # MIL seting parameters
    parser.add_argument('--patch_size', default=32, type=int, help='patch size to split image')  # 16/32/48/64/96/128

    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    parser.add_argument('--num_epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--intake_epochs', default=0, type=int, help='only save model at epochs after intake_epochs')
    parser.add_argument('--lr', default=0.00001, type=float, help='learing rate')
    parser.add_argument('--lrf', type=float, default=0.,
                        help='learing rate decay rate, default 0(not enabled), suggest 0.05/0.01/0.2/0.1')

    # head_balance weight
    parser.add_argument('--CLS_MIL_head_weight', default=1., type=float,
                        help='balance weight for MIL_head in non-shuffle step (CLS step)')
    parser.add_argument('--MIL_head_weight', default=1., type=float,
                        help='balance weight for MIL_head in shuffle step (MIL step)')

    parser.add_argument('--opt_name', default='Adam', type=str, help='optimizer name Adam or SGD')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
