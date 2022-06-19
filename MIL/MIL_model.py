"""
MIL-SI model   Script  ver： Jun 10th 11:20

TODO support for Swin Transformer

TODO  MIL_pooling

ref:
arxiv 2006.01561 official code

distribution pooling mudule from
https://github.com/onermustafaumit/mil_pooling_filters/blob/main/regression/distribution_pooling_filter.py
"""
import math
import timm
import torch
import torch.nn as nn
from MIL import MIL_Transformer_blocks, MIL_ResNet_blocks


class DistributionPoolingFilter(nn.Module):
    __constants__ = ['num_bins', 'sigma']

    def __init__(self, num_bins=1, sigma=0.1):
        super(DistributionPoolingFilter, self).__init__()

        self.num_bins = num_bins
        self.sigma = sigma
        self.alfa = 1 / math.sqrt(2 * math.pi * (sigma ** 2))
        self.beta = -1 / (2 * (sigma ** 2))

        sample_points = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sample_points', sample_points)

    def extra_repr(self):
        return 'num_bins={}, sigma={}'.format(
            self.num_bins, self.sigma
        )

    def forward(self, data):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


class DistributionWithAttentionPoolingFilter(DistributionPoolingFilter):

    def __init__(self, num_bins=1, sigma=0.1):
        super(DistributionWithAttentionPoolingFilter, self).__init__(num_bins, sigma)

    def forward(self, data, attention_weights):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        # attention_weights.size() --> (batch_size,num_instances)
        attention_weights = torch.reshape(attention_weights, (batch_size, num_instances, 1, 1))
        # attention_weights.size() --> (batch_size,num_instances,1,1)
        attention_weights = attention_weights.repeat(1, 1, num_features, self.num_bins)
        # attention_weights.size() --> (batch_size,num_instances,num_features,num_bins)

        result = attention_weights * result
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


# represtation_MLP  (FFN like)
class represtation_MLP(nn.Module):
    """
    FFN
    todo  MIL_pooling=DistributionPoolingFilter() 降低内存开销，引入MIL模块
    input size of [bag_num,channel,H,W]
    output size of [1,1]

    :param in_features: input neurons
    :param hidden_features: hidden neurons
    :param out_features: output neurons
    :param act_layer: nn.GELU
    :param drop: last drop
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.in_features = in_features

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.hidden_features)

        self.fc3 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if len(x.shape) == 3:  # Transformer input
            batch, num_patches, feature_dim = x.shape
            x = x.view(batch, -1)  # batch, num_patches * feature_dim

        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.drop(x)

        return x


class CLS_SwinTransformer_model(nn.Module):
    def __init__(self, MIL_SwinTransformer_model):
        super().__init__()
        self.backbone = MIL_SwinTransformer_model.backbone
        self.head = MIL_SwinTransformer_model.head

    def forward(self, x):
        x = self.backbone.forward_features(x)
        cls = self.head(x)
        return cls


class MIL_SwinTransformer_model(nn.Module):

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.num_features = backbone.num_features
        self.num_classes = num_classes
        # Transformer backbone
        self.backbone = backbone
        self.num_patches = self.backbone.patch_embed.num_patches

        # MIL represtation head
        self.MIL_MLP = represtation_MLP(self.num_features,
                                        self.num_features // 2,  # todo 未来优化一下这一块
                                        1 + self.num_classes)
        '''
        self.MIL_MLP = nn.Linear(self.num_features, 1 + self.num_classes)
        '''
        # Classifier head
        self.head = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x, ask_CLS_head=False):
        x = self.backbone.forward_features(x)  # (batch, num_patches, feature_dim)

        # (batch, feature_dim) -> (batch, 1 + self.num_classes)  soft_label regression output
        MIL_x = self.MIL_MLP(x)

        if ask_CLS_head:  # get CLS head output
            # (batch, feature_dim) -> (batch, self.num_classes)  one-hot
            cls = self.head(x)
            return MIL_x, cls
        else:
            return MIL_x

    def Stripe(self):
        Stripe_model = CLS_SwinTransformer_model(self)
        return Stripe_model


class CLS_Transformer_model(nn.Module):
    def __init__(self, MIL_Transformer_model):
        super().__init__()
        self.backbone = MIL_Transformer_model.backbone
        self.head = MIL_Transformer_model.head

    def forward(self, x):
        x = self.backbone(x)
        cls_token = x[:, 0]  # (batch, feature_dim)
        cls = self.head(cls_token)
        return cls


class MIL_Transformer_model(nn.Module):

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.num_features = backbone.num_features
        self.num_classes = num_classes
        # Transformer backbone
        self.backbone = backbone
        self.num_patches = self.backbone.patch_embed.num_patches

        # MIL represtation head
        self.MIL_MLP = represtation_MLP(self.num_features * self.num_patches,
                                        self.num_patches // 2,  # todo 未来优化一下这一块
                                        1 + self.num_classes)
        '''
        self.MIL_MLP = nn.Linear(self.num_features, 1 + self.num_classes)
        '''
        # Classifier head
        self.head = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x, ask_CLS_head=False):
        x = self.backbone(x)

        cls_token = x[:, 0]  # (batch, feature_dim)
        patche_tokens = x[:, 1:]  # (batch, num_patches, feature_dim)

        # (batch, num_patches, feature_dim) -> (batch, 1 + self.num_classes)  soft_label regression output
        x = self.MIL_MLP(patche_tokens)
        # x = self.MIL_MLP(cls_token)

        if ask_CLS_head:  # get CLS head output
            # (batch, feature_dim) -> (batch, self.num_classes)  one-hot
            cls = self.head(cls_token)
            return x, cls
        else:
            return x

    def Stripe(self):
        Stripe_model = CLS_Transformer_model(self)
        return Stripe_model


class CLS_ResNet_model(nn.Module):
    def __init__(self, MIL_ResNet_model):
        super().__init__()
        self.backbone = MIL_ResNet_model.backbone
        self.avgpool = MIL_ResNet_model.avgpool
        self.head = MIL_ResNet_model.head

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        cls = self.head(x)
        return cls


class MIL_ResNet_model(nn.Module):

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.num_features = backbone.num_features
        self.num_classes = num_classes
        # ResNet backbone
        self.backbone = backbone
        self.avgpool = backbone.avgpool

        # MIL represtation head
        self.MIL_MLP = represtation_MLP(self.num_features,
                                        self.num_features // 2,  # todo 未来优化一下这一块
                                        1 + self.num_classes)
        '''
        self.MIL_MLP = nn.Linear(self.num_features, 1 + self.num_classes)
        '''
        # Classifier head
        self.head = nn.Linear(self.num_features, self.num_classes)

    def forward(self, x, ask_CLS_head=False):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        MIL_x = self.MIL_MLP(x)

        if ask_CLS_head:  # get CLS head output
            # (batch, feature_dim) -> (batch, self.num_classes)  one-hot
            cls = self.head(x)
            return MIL_x, cls
        else:
            return MIL_x

    def Stripe(self):
        Stripe_model = CLS_ResNet_model(self)
        return Stripe_model


def build_backbone(model_idx='ViT_b', img_size=224, pretrained_backbone=True):

    if model_idx[0:5] == 'ViT_t':  # vit_tiny
        model = MIL_Transformer_blocks.VisionTransformer(img_size, patch_size=16, in_chans=3, num_classes=1000,
                                                         embed_dim=192, depth=12,
                                                         num_heads=3, mlp_ratio=4.)
        if img_size == 224:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_tiny_patch16_224', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        elif img_size == 384:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_tiny_patch16_384', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        else:
            print('not a avaliable image size with', model_idx)
            raise

    elif model_idx[0:5] == 'ViT_s':  # vit_small
        model = MIL_Transformer_blocks.VisionTransformer(img_size, patch_size=16, in_chans=3, num_classes=1000,
                                                         embed_dim=384, depth=12,
                                                         num_heads=6, mlp_ratio=4.)
        if img_size == 224:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_small_patch16_224', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        elif img_size == 384:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_small_patch16_384', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        else:
            print('not a avaliable image size with', model_idx)
            raise

    elif model_idx[0:5] == 'ViT_l':  # vit_large
        model = MIL_Transformer_blocks.VisionTransformer(img_size, patch_size=16, in_chans=3, num_classes=1000,
                                                         embed_dim=1024, depth=24,
                                                         num_heads=16, mlp_ratio=4.)
        if img_size == 224:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_large_patch16_224', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        elif img_size == 384:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_large_patch16_384', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        else:
            print('not a avaliable image size with', model_idx)
            raise

    elif model_idx[0:5] == 'ViT_h':  # vit_huge
        model = MIL_Transformer_blocks.VisionTransformer(img_size, patch_size=14, in_chans=3, num_classes=21843,
                                                         embed_dim=1280, depth=32,
                                                         num_heads=16, mlp_ratio=4.)
        if img_size == 224:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_huge_patch14_224_in21k', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        else:
            print('not a avaliable image size with', model_idx)
            raise

    elif model_idx[0:5] == 'ViT_b' or model_idx[0:3] == 'ViT':  # vit_base
        model = MIL_Transformer_blocks.VisionTransformer(img_size, patch_size=16, in_chans=3, num_classes=1000,
                                                         embed_dim=768, depth=12,
                                                         num_heads=12, mlp_ratio=4.)
        if img_size == 224:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        elif img_size == 384:
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_base_patch16_384', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        else:
            print('not a avaliable image size with', model_idx)
            raise

    elif model_idx[0:6] == 'ResNet':  # same model for 224 or 384 img
        model = MIL_ResNet_blocks.build_ResNet50_backbone(edge_size=img_size, pretrained=pretrained_backbone)
        return model

    elif model_idx[0:6] == 'swin_b':  # same model for 224 or 384 img
        if img_size == 224:
            # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained_backbone,
                                      num_classes=1000)
            return model

        elif img_size == 384:
            # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
            model = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained_backbone,
                                      num_classes=1000)
            return model
        else:
            print('not a avaliable image size with', model_idx)
            raise

    else:
        print('not a avaliable MIL backbone model')
        raise


def build_MIL_model(model_idx, edge_size, pretrained_backbone=True, num_classes=1000):
    # build backbone
    backbone = build_backbone(model_idx, edge_size, pretrained_backbone)

    # build MIL enhanced model
    if model_idx[0:3] == 'ViT':
        model = MIL_Transformer_model(backbone, num_classes)
        return model

    elif model_idx[0:8] == 'ResNet50':
        model = MIL_ResNet_model(backbone, num_classes)
        return model

    elif model_idx[0:6] == 'swin_b':
        model = MIL_SwinTransformer_model(backbone, num_classes)
        return model

    else:
        print('not a valid MIL model')
        return -1


'''
# cuda issue
print('cuda avaliablity:', torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# play information
img_size = 384
num_classes = 2

# play data
x = torch.randn(1, 3, img_size, img_size).to(dev)

# Model
model = build_MIL_model(model_idx='ViT_l_384_401_PT_lf05_b4_ROSE_MIL',
                        edge_size=img_size, pretrained_backbone=True, num_classes=num_classes)
model.to(dev)

# MIL Train
y = model(x)
print('MIL Train', y.shape)

print('')

# CLS Train
y, cls = model(x, True)
print('MIL Train', y.shape)
print('CLS Train (one hot)', cls.shape)
print('')

# Stripe
Stripe_model = model.Stripe()
y = Stripe_model(x)
print('Stripe Model', y.shape)
'''
