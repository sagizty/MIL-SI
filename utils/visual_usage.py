"""
Attention Visulization    Script  ver： Nov 3rd 19:30
use rgb format input
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision.transforms import ToPILImage


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    sof = nn.Softmax()
    return sof(x)


def imshow(inp, title=None):  # Imshow for Tensor
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    '''
    # if required: Alter the transform 
    # because transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    '''
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Grad CAM part：Visualize of CNN+Transformer attention area
def cls_token_s12_transform(tensor, height=12, width=12):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def cls_token_s14_transform(tensor, height=14, width=14):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def cls_token_s16_transform(tensor, height=16, width=16):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def cls_token_s24_transform(tensor, height=24, width=24):  # based on pytorch_grad_cam
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def no_cls_token_s12_transform(tensor, height=12, width=12):  # based on pytorch_grad_cam
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swinT_transform_224(tensor, height=7, width=7):  # 224 7
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swinT_transform_384(tensor, height=12, width=12):  # 384 12
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def choose_cam_by_model(model, model_idx, edge_size, use_cuda=True, model_type='CLS'):
    """
    :param model: model object
    :param model_idx: model idx for the getting pre-setted layer and size
    :param edge_size: image size for the getting pre-setted layer and size

    :param use_cuda: use cuda to speed up imaging
    :param model_type: default 'CLS' for model, 'MIL' for model backbone
    """
    from pytorch_grad_cam import GradCAM

    # reshape_transform  todo conformer 224!!
    # check class: target_category = None
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.

    if model_idx[0:3] == 'ViT' or model_idx[0:4] == 'deit':
        # We should chose any layer before the final attention block,
        # check: https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
        if model_type == 'CLS':
            target_layers = [model.blocks[-1].norm1]
        else:  # MIL-SI
            target_layers = [model.backbone.blocks[-1].norm1]

        if model_idx[0:5] == 'ViT_h':
            if edge_size == 224:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s16_transform)
            else:
                print('ERRO in ViT_huge edge size')
                return -1
        else:
            if edge_size == 384:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s24_transform)
            elif edge_size == 224:
                grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                                   reshape_transform=cls_token_s14_transform)
            else:
                print('ERRO in ViT/DeiT edge size')
                return -1

    elif model_idx[0:3] == 'vgg':
        if model_type == 'CLS':
            target_layers = [model.features[-1]]
        else:
            target_layers = [model.backbone.features[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)

    elif model_idx[0:6] == 'swin_b':
        if model_type == 'CLS':
            target_layers = [model.layers[-1].blocks[-1].norm1]
        else:
            target_layers = [model.backbone.layers[-1].blocks[-1].norm1]
        if edge_size == 384:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=swinT_transform_384)
        elif edge_size == 224:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=swinT_transform_224)
        else:
            print('ERRO in Swin Transformer edge size')
            return -1

    elif model_idx[0:6] == 'ResNet':
        if model_type == 'CLS':
            target_layers = [model.layer4[-1]]
        else:
            target_layers = [model.backbone.layer4[-1]]

        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None

    elif model_idx[0:7] == 'Hybrid1' and edge_size == 384:
        target_layers = [model.blocks[-1].norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s12_transform)

    elif model_idx[0:7] == 'Hybrid2' and edge_size == 384:
        target_layers = [model.dec4.norm1]

        if 'CLS' in model_idx.split('_') and 'No' in model_idx.split('_'):
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=no_cls_token_s12_transform)

        else:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s12_transform)

    elif model_idx[0:7] == 'Hybrid3' and edge_size == 384:
        target_layers = [model.dec3.norm1]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                           reshape_transform=cls_token_s24_transform)

    elif model_idx[0:9] == 'mobilenet':
        if model_type == 'CLS':
            target_layers = [model.blocks[-1]]
        else:
            target_layers = [model.backbone.blocks[-1]]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None

    elif model_idx[0:10] == 'ResN50_ViT' and edge_size == 384:
        if model_type == 'CLS':
            target_layers = [model.blocks[-1].norm1]
        else:
            target_layers = [model.backbone.blocks[-1].norm1]
        if edge_size == 384:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s24_transform)
        elif edge_size == 224:
            grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda,
                               reshape_transform=cls_token_s14_transform)
        else:
            print('ERRO in ResN50_ViT edge size')
            return -1

    elif model_idx[0:12] == 'efficientnet':
        target_layers = [model.conv_head]
        grad_cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=None)  # CNN: None


    else:
        print('ERRO in model_idx')
        return -1

    return grad_cam


def check_SAA(inputs, labels, model, model_idx, edge_size, class_names, model_type='CLS', num_images=-1, pic_name='test',
              draw_path='../imaging_results', check_all=True, unknown_GT=False, writer=None):
    """
    check num_images of images and visual the models's attention area
    output a pic with 2 column and rows of num_images

    :param inputs: inputs of data
    :param labels: labels or the K+1 soft label of data

    :param model: model object
    :param model_idx: model idx for the getting pre-setted layer and size
    :param edge_size: image size for the getting pre-setted layer and size

    :param class_names: The name of classes for painting
    :param model_type: default 'CLS' for model, 'MIL' for model backbone

    :param num_images: how many image u want to check, should SMALLER THAN the batchsize
    :param pic_name: name of the output pic
    :param draw_path: path folder for output pic
    :param check_all: choose the type of checking CAM : by default False to be only on the predicted type'
                    True to be on all types

    :param unknown_GT: cam on unknown GT

    :param writer: attach the pic to the tensorboard backend

    :return: None
    """
    from pytorch_grad_cam.utils import show_cam_on_image

    # choose checking type: false to be only on the predicted type'; true to be on all types
    if check_all:
        checking_type = ['ori', ]
        checking_type.extend([cls for cls in range(len(class_names))])
    else:
        checking_type = ['ori', 'tar']

    # test model
    was_training = model.training
    model.eval()

    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    grad_cam = choose_cam_by_model(model, model_idx, edge_size, model_type=model_type)  # choose model

    if num_images == -1:  # auto detect a batch
        num_images = int(inputs.shape[0])

    images_so_far = 0
    plt.figure()

    for j in range(num_images):

        for type in checking_type:
            images_so_far += 1
            if type == 'ori':
                ax = plt.subplot(num_images, len(checking_type), images_so_far)
                ax.axis('off')

                if unknown_GT and not len(labels) == 1:  # Ground Truth of the K+1 soft label
                    soft_label = labels.cpu().numpy()[j]  # K+1 soft label
                    title = 'A' + str(round(soft_label[0], 0))
                    for i in range(1, len(soft_label)):
                        title += class_names[i - 1][0]  # use the first character only
                        title += str(round(soft_label[i], 0))  # use int (float 0)
                        title += ' '
                    ax.set_title(title)

                else:
                    ax.set_title('Ground Truth:{}'.format(class_names[int(labels[j])]))

                imshow(inputs.cpu().data[j])
                plt.pause(0.001)  # pause a bit so that plots are updated

            else:
                ax = plt.subplot(num_images, len(checking_type), images_so_far)
                ax.axis('off')
                if type == 'tar':
                    ax.set_title('Predict: {}'.format(class_names[preds[j]]))
                    # focus on the specific target class to create grayscale_cam
                    # grayscale_cam is generate on batch
                    grayscale_cam = grad_cam(inputs, target_category=None, eigen_smooth=False, aug_smooth=False)
                else:
                    # pseudo confidence by softmax
                    ax.set_title('{:.1%} {}'.format(softmax(outputs[j])[int(type)], class_names[int(type)]))
                    # focus on the specific target class to create grayscale_cam
                    # grayscale_cam is generate on batch
                    grayscale_cam = grad_cam(inputs, target_category=int(type), eigen_smooth=False, aug_smooth=False)

                # get a cv2 encoding image from dataloder by inputs[j].cpu().numpy().transpose((1, 2, 0))

                cam_img = show_cam_on_image(inputs[j].cpu().numpy().transpose((1, 2, 0)), grayscale_cam[j],
                                            use_rgb=True)  # Fixme: use rgb format input (already fixed)

                plt.imshow(cam_img)
                plt.pause(0.001)  # pause a bit so that plots are updated

            if images_so_far == num_images * len(checking_type):  # complete when the pics is enough
                picpath = os.path.join(draw_path, pic_name + '.jpg')
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                plt.savefig(picpath, dpi=1000)
                plt.show()

                model.train(mode=was_training)
                if writer is not None:  # attach the pic to the tensorboard backend if avilable
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                plt.cla()
                plt.close("all")
                return

    model.train(mode=was_training)


def visualize_check(inputs, labels, model, class_names, num_images=-1, pic_name='test',
                    draw_path='/home/ZTY/imaging_results', writer=None):  # visual check
    """
    check num_images of images and visual them
    output a pic with 3 column and rows of num_images//3

    :param inputs: inputs of data
    :param labels: labels of data

    :param model: model object
    :param class_names: The name of classes for painting
    :param num_images: how many image u want to check, should SMALLER THAN the batchsize
    :param pic_name: name of the output pic
    :param draw_path: path folder for output pic
    :param writer: attach the pic to the tensorboard backend

    :return:  None

    """
    was_training = model.training
    model.eval()

    images_so_far = 0
    plt.figure()

    with torch.no_grad():

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if num_images == -1:  # auto detect a batch
            num_images = int(inputs.shape[0])

        if num_images % 5 == 0:
            line_imgs_num = 5
        elif num_images % 4 == 0:
            line_imgs_num = 4
        elif num_images % 3 == 0:
            line_imgs_num = 3
        elif num_images % 2 == 0:
            line_imgs_num = 2
        else:
            line_imgs_num = int(num_images)

        rows_imgs_num = int(num_images // line_imgs_num)
        num_images = line_imgs_num * rows_imgs_num

        for j in range(num_images):  # each batch input idx: j

            images_so_far += 1

            ax = plt.subplot(rows_imgs_num, line_imgs_num, images_so_far)

            ax.axis('off')
            ax.set_title('Pred: {} True: {}'.format(class_names[preds[j]], class_names[int(labels[j])]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                picpath = os.path.join(draw_path, pic_name + '.jpg')
                if not os.path.exists(draw_path):
                    os.makedirs(draw_path)

                '''
                myfig = plt.gcf()  # get current image
                myfig.savefig(picpath, dpi=1000)
                '''
                plt.savefig(picpath, dpi=1000)
                plt.show()

                model.train(mode=was_training)

                if writer is not None:  # attach the pic to the tensorboard backend if avilable
                    image_PIL = Image.open(picpath)
                    img = np.array(image_PIL)
                    writer.add_image(pic_name, img, 1, dataformats='HWC')

                plt.cla()
                plt.close("all")
                return

        model.train(mode=was_training)


def unpatchify(pred, patch_size=16):
    """
    Decoding embeded patch tokens

    input:
    x: (B, num_patches, patch_size**2 *3) AKA [B, num_patches, flatten_dim]
    patch_size:

    output:
    imgs: (B, 3, H, W)
    """

    # squre root of num_patches (without CLS token is required)
    h = w = int(pred.shape[1] ** .5)
    # assert num_patches is with out CLS token
    assert h * w == pred.shape[1]

    # ReArrange dimensions [B, num_patches, flatten_dim] -> [B, h_p, w_p, patch_size, patch_size, C]
    pred = pred.reshape(shape=(pred.shape[0], h, w, patch_size, patch_size, 3))
    # ReArrange dimensions [B, h_p, w_p, patch_size, patch_size, C] -> [B, C, h_p, patch_size, w_p, patch_size]
    pred = torch.einsum('nhwpqc->nchpwq', pred)
    # use reshape to compose patch [B, C, h_p, patch_size, w_p, patch_size] -> [B, C, H, W]
    imgs = pred.reshape(shape=(pred.shape[0], 3, h * patch_size, h * patch_size))
    return imgs


def patchify(imgs, patch_size=16):
    """
    Break image to patch tokens

    input:
    imgs: (B, 3, H, W)

    output:
    x: (B, num_patches, patch_size**2 *3) AKA [B, num_patches, flatten_dim]
    """
    # assert H == W and image shape is dividedable by patch
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
    # patch num in rol or column
    h = w = imgs.shape[2] // patch_size

    # use reshape to split patch [B, C, H, W] -> [B, C, h_p, patch_size, w_p, patch_size]
    imgs = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))

    # ReArrange dimensions [B, C, h_p, patch_size, w_p, patch_size] -> [B, h_p, w_p, patch_size, patch_size, C]
    imgs = torch.einsum('nchpwq->nhwpqc', imgs)
    # ReArrange dimensions [B, h_p, w_p, patch_size, patch_size, C] -> [B, num_patches, flatten_dim]
    imgs = imgs.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 3))
    return imgs


def anti_tensor_norm(batch_tensor):
    pass  # TODO 总之想一下
