"""
todo  MIL_pooling
MIL-SI model   Script  ver： Apr 23th 22:00

ref:
arxiv 2006.01561 official code

distribution pooling mudule from
https://github.com/onermustafaumit/mil_pooling_filters/blob/main/regression/distribution_pooling_filter.py
"""
import math
import timm
import torch
import torch.nn as nn
from MIL import MIL_Transformer_blocks


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
        batch, num_patches, feature_dim = x.shape
        x = x.view(batch, -1)  # batch, num_patches * feature_dim

        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.drop(x)

        return x


def build_backbone(model_idx='ViT', img_size=224, pretrained_backbone=True):
    if model_idx[0:3] == 'ViT':
        if img_size == 224:
            model = MIL_Transformer_blocks.VisionTransformer(img_size)
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        elif img_size == 384:
            model = MIL_Transformer_blocks.VisionTransformer(img_size)
            if pretrained_backbone:
                backbone_weights = timm.create_model('vit_base_patch16_384', pretrained=True).state_dict()
                model.load_state_dict(backbone_weights, False)
            return model

        else:
            print('not a avaliable image size')
            raise
    else:
        print('not a avaliable MIL backbone model')
        raise


class CLS_model(nn.Module):
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
        Stripe_model = CLS_model(self)
        return Stripe_model


'''
# cuda issue
print('cuda avaliablity:', torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# play information
img_size = 224
num_classes = 2

# play data
x = torch.randn(1, 3, img_size, img_size).to(dev)

# Model
backbone = build_backbone(model_idx='ViT', img_size=img_size)
model = MIL_Transformer_model(backbone, num_classes)
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
