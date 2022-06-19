"""
MIL Testing   Script  ver： Jun 5th 21:00
"""
from __future__ import print_function, division
import json
import time
import argparse
from tensorboardX import SummaryWriter

from utils.visual_usage import *
from utils.tools import del_file

from MIL.MIL_model import *
from MIL.MIL_structure import *


def test_model(model, shuffle_patch_distributer, fixed_patch_distributer, test_dataloader, MIL_criterion, CLS_criterion,
               class_names, test_dataset_size, edge_size=384, CLS_MIL=True, model_idx=None, test_model_idx=None,
               check_minibatch=100, head_balance=(1., 1., 1.), device=None, draw_path='../runs',
               enable_attention_check=False, enable_visualize_check=False, shuffle_attention_check=False,
               MIL_Stripe=False, writer=None):

    if shuffle_patch_distributer is None:
        shuffle_MIL = False
    else:
        shuffle_MIL = True

    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    print('Epoch: Test')
    print('-' * 10)

    phase = 'test'
    index = 0
    model_time = time.time()

    # initiate the empty json dict
    json_log = {'test': {}}

    # initiate the empty log dict
    log_dict = {}
    for cls_idx in range(len(class_names)):
        log_dict[class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

    model.eval()  # Set model to evaluate mode

    # criterias, initially empty
    running_loss = 0.0
    log_running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for patch_image, patch_labels, labels in test_dataloader:  # use different dataloder in different phase
        # get datas
        patch_image = patch_image.to(device)
        patch_labels = patch_labels.to(device)
        labels = labels.to(device)

        # MIL-SI forward with 2-step(shuffle step + non-shuffle step)

        if not MIL_Stripe:
            # get bag data from the distributers
            if shuffle_MIL:
                MIL_bag_image, MIL_bag_labels = shuffle_patch_distributer(patch_image, patch_labels)
            CLS_bag_image, CLS_bag_labels, labels = fixed_patch_distributer(patch_image, patch_labels, labels)

            # zero the parameter gradients only need in training
            # optimizer.zero_grad()

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

        else:
            CLS_bag_image, CLS_bag_labels, labels = fixed_patch_distributer(patch_image, patch_labels, labels)

            if shuffle_attention_check:
                if shuffle_MIL:
                    MIL_bag_image, MIL_bag_labels = shuffle_patch_distributer(patch_image, patch_labels)
                else:
                    print('no MIL_patch_distributer -> no shuffle_attention_check')
            # zero the parameter gradients only need in training
            # optimizer.zero_grad()
            # forward
            outputs = model(CLS_bag_image)
            _, preds = torch.max(outputs, 1)
            loss = CLS_criterion(outputs, labels)

        # log criterias: update
        log_running_loss += loss.item()
        running_loss += loss.item() * patch_image.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Compute recision and recall for each class.
        for cls_idx in range(len(class_names)):
            # NOTICE remember to put tensor back to cpu
            tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                        (preds == cls_idx).cpu().numpy().astype(int))
            tn = np.dot((labels.cpu().data != cls_idx).numpy().astype(int),
                        (preds != cls_idx).cpu().numpy().astype(int))

            fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

            fn = np.sum((labels.cpu().data == cls_idx).numpy()) - tp

            # log_dict[cls_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            log_dict[class_names[cls_idx]]['tp'] += tp
            log_dict[class_names[cls_idx]]['tn'] += tn
            log_dict[class_names[cls_idx]]['fp'] += fp
            log_dict[class_names[cls_idx]]['fn'] += fn

        # attach the records to the tensorboard backend
        if writer is not None:
            # ...log the running loss
            writer.add_scalar(phase + ' minibatch loss',
                              float(loss.item()),
                              index)
            writer.add_scalar(phase + ' minibatch ACC',
                              float(torch.sum(preds == labels.data) / patch_image.size(0)),
                              index)

        # at the checking time now
        if index % check_minibatch == check_minibatch - 1:
            model_time = time.time() - model_time

            check_index = index // check_minibatch + 1

            epoch_idx = 'test'
            print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                  check_index, '     time used:', model_time)

            print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

            # how many image u want to check, should SMALLER THAN the batchsize

            if enable_attention_check:
                try:
                    if not MIL_Stripe:
                        check_SAA(CLS_bag_image, labels, model, model_idx, edge_size, class_names, model_type='MIL',
                                  num_images=1, pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                                  draw_path=draw_path, writer=writer)
                    else:
                        check_SAA(CLS_bag_image, labels, model, model_idx, edge_size, class_names, model_type='MIL',
                                  num_images=1, pic_name='Stripe_GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                                  draw_path=draw_path, writer=writer)
                except:
                    print('model:', model_idx, ' with edge_size', edge_size, 'is not supported yet')
            else:
                pass

            if shuffle_attention_check and shuffle_MIL:
                if len(labels) > 1:  # batch size > 1
                    check_labels = MIL_bag_labels  # size of [B, K+1]
                else:
                    check_labels = labels  # a long tensor size of the batch, each label is a catalog idx

                try:
                    if not MIL_Stripe:
                        check_SAA(MIL_bag_image, check_labels, model, model_idx, edge_size, class_names,
                                  model_type='MIL', num_images=1,
                                  pic_name='shuffle_GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                                  draw_path=draw_path, unknown_GT=True, writer=writer)
                    else:
                        check_SAA(MIL_bag_image, check_labels, model, model_idx, edge_size, class_names,
                                  model_type='MIL', num_images=1,
                                  pic_name='shuffle_Stripe_GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                                  draw_path=draw_path, unknown_GT=True, writer=writer)
                except:
                    print('model:', model_idx, ' with edge_size', edge_size, 'is not supported yet')
            else:
                pass

            if enable_visualize_check:
                visualize_check(CLS_bag_image, labels, model, class_names, num_images=3,
                                pic_name='Visual_' + str(epoch_idx) + '_I_' + str(index + 1),
                                draw_path=draw_path, writer=writer)

            model_time = time.time()
            log_running_loss = 0.0

        index += 1
    # json log: update
    json_log['test'][phase] = log_dict

    # log criterias: print
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size * 100
    print('\nEpoch:  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    for cls_idx in range(len(class_names)):
        # calculating the confusion matrix
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

    print('\n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # save json_log  indent=2 for better view
    json.dump(json_log, open(os.path.join(draw_path, test_model_idx + '_log.json'), 'w'), ensure_ascii=False, indent=2)

    return model


def main(args):
    if args.paint:
        # use Agg kernal, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    enable_notify = args.enable_notify  # False
    enable_tensorboard = args.enable_tensorboard  # False

    enable_attention_check = args.enable_attention_check  # False
    enable_visualize_check = args.enable_visualize_check  # False
    shuffle_attention_check = args.shuffle_attention_check  # False

    model_idx = args.model_idx  # the model we are going to use. by the format of Model_size_other_info

    # shuffle_dataloader
    shuffle_dataloader = True if args.shuffle_dataloader else False  # False
    # shuffle MIL approach
    shuffle_MIL = False if args.shuffle_MIL_off else True  # True
    # CLS step MIL
    CLS_MIL = False if args.CLS_MIL_off else True  # True

    MIL_Stripe = args.MIL_Stripe  # remove MIL head and analysis the CLS branch
    # fixme why when MIL head is used, CAM is wrong? Youdan Feng: ..
    if not MIL_Stripe:
        enable_attention_check = False
        shuffle_attention_check = False

    # PATH info
    draw_root = args.draw_root
    model_path = args.model_path
    dataroot = args.dataroot

    data_augmentation_mode = args.data_augmentation_mode  # 0

    # image size and patch_size for the input MIL image
    edge_size = args.edge_size  # 224 384 1000
    patch_size = args.patch_size

    batch_size = args.batch_size  # GPU memory cost: colab 4  gpu server 8

    head_balance = (1., 1. * args.CLS_MIL_head_weight, 1. * args.MIL_head_weight)

    # skip minibatch
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 80 // batch_size

    # Format experiment record name
    if shuffle_attention_check:
        if not MIL_Stripe:
            test_model_idx = 'shuffle_attention_check_MIL_' + model_idx
        else:
            test_model_idx = 'shuffle_attention_check_MIL_Stripe_' + model_idx
    else:
        if not MIL_Stripe:
            test_model_idx = 'MIL_' + model_idx
        else:
            test_model_idx = 'MIL_Stripe_' + model_idx

    if shuffle_dataloader:
        test_model_idx += '_shuffle_dataloader'

    test_model_idx += '_b_' + str(batch_size) + '_test'

    # PATH
    draw_path = os.path.join(draw_root, test_model_idx)
    save_model_path = os.path.join(model_path, 'MIL_' + model_idx + '.pth')

    # choose the test dataset
    test_dataroot = os.path.join(dataroot, 'test')

    if os.path.exists(draw_path):
        del_file(draw_path)  # clear the output folder, NOTICE this may be DANGEROUS
    else:
        os.makedirs(draw_path)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(draw_path)
    else:
        writer = None

    print("*********************************{}*************************************".format('setting'))
    print(args)

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
                raise
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
                raise
    print('GPU:', gpu_use)

    # device enviorment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_dataset and its info
    test_dataset = MILDataset(test_dataroot, mode='val', data_augmentation_mode=data_augmentation_mode,
                              suffix='.jpg', edge_size=edge_size, patch_size=patch_size)
    class_names = test_dataset.class_names
    test_dataset_size = len(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_dataloader,
                                                  num_workers=1)

    if enable_notify:  # use notifyemail to send the record to somewhere
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='xxxx@163.com', mail_pass='xxxx',
                       default_reciving_list=['xxxx@163.com'],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('testing model_idx: ' + str(model_idx) + '.update to the tensorboard')
        else:
            notify.add_text('testing model_idx: ' + str(model_idx) + '.not update to the tensorboard')
        notify.add_text('  ')
        notify.add_text('GPU use: ' + str(gpu_use))
        notify.add_text('  ')
        notify.add_text('cls number ' + str(len(class_names)))
        notify.add_text('edge size ' + str(edge_size))
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('MIL_patch_size ' + str(patch_size))
        notify.send_log()

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
    model = build_MIL_model(model_idx, edge_size, pretrained_backbone=False, num_classes=len(class_names))

    # get Pre_Trained model if required
    try:
        model.load_state_dict(torch.load(save_model_path))
        print("model loaded")
        if MIL_Stripe:
            model = model.Stripe()
        print("model :", model_idx)
    except:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(save_model_path), False)
            if MIL_Stripe:
                model = model.Stripe()
            print("DataParallel model loaded")
            print("model :", model_idx)
        except:
            print("model loading erro!!")
            return -1

    # put on multi-gpu
    if gpu_use == -1:
        model = nn.DataParallel(model)
    model.to(device)

    # criterion setting
    MIL_criterion = torch.nn.L1Loss(size_average=None, reduce=None)  # todo find better loss, maybe smoothl1loss？
    CLS_criterion = nn.CrossEntropyLoss()

    test_model(model, shuffle_patch_distributer, fixed_patch_distributer, test_dataloader, MIL_criterion, CLS_criterion,
               class_names, test_dataset_size, edge_size=edge_size, CLS_MIL=CLS_MIL, model_idx=model_idx,
               test_model_idx=test_model_idx, check_minibatch=check_minibatch, head_balance=head_balance,
               device=device, draw_path=draw_path, enable_attention_check=enable_attention_check,
               enable_visualize_check=enable_visualize_check, shuffle_attention_check=shuffle_attention_check,
               MIL_Stripe=MIL_Stripe, writer=writer)


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_idx', default='ViT_384_401_PT_lf05_b4_ROSE_MIL', type=str, help='Model Name or index')

    # shuffle_MIL, set --shuffle_MIL_off to use only the non-shuffle step to train 2-head, by default use both steps
    parser.add_argument('--shuffle_MIL_off', action='store_true', help='disable shuffle MIL training')
    # no-shuffle step (CLS step) MIL head, set --CLS_MIL_off to use only CLS head in the non-shuffle step
    parser.add_argument('--CLS_MIL_off', action='store_true', help='disable CLS_MIL training')

    # shuffle_dataloader
    parser.add_argument('--shuffle_dataloader', action='store_true', help='shuffle Test dataset')

    # MIL Stripe
    parser.add_argument('--MIL_Stripe', action='store_true', help='MIL_Stripe')

    # Trained models
    # '/home/MIL_Experiment/saved_models/PC_Hybrid2_384_PreTrain_000.pth'
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--dataroot', default='/data/MIL_Experiment/dataset/ROSE_MIL',
                        help='path to dataset')
    parser.add_argument('--model_path', default='/home/MIL_Experiment/saved_models',
                        help='path to state-dicts of the saved models')
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
    parser.add_argument('--shuffle_attention_check', action='store_true',
                        help='check and save attention map on shuffle images')

    # Dataset based parameters
    parser.add_argument('--edge_size', default=384, type=int, help='edge size of input image')  # 224 256 384 1000

    # Training seting parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Testing batch_size default 1')

    # MIL seting parameters
    parser.add_argument('--patch_size', default=32, type=int, help='patch size to split image')  # 16/32/48/64/96/128

    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    # head_balance weight
    parser.add_argument('--CLS_MIL_head_weight', default=1., type=float,
                        help='balance weight for MIL_head in non-shuffle step (CLS step)')
    parser.add_argument('--MIL_head_weight', default=1., type=float,
                        help='balance weight for MIL_head in shuffle step (MIL step)')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
