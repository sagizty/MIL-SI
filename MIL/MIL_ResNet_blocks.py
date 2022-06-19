"""
ResNet Blocks  Script  ver： Jun 6th 14:50

ResNet stages' feature map

# input = 3, 384, 384
torch.Size([1, 256, 96, 96])
torch.Size([1, 512, 48, 48])
torch.Size([1, 1024, 24, 24])
torch.Size([1, 2048, 12, 12])
torch.Size([1, 1000])

# input = 3, 224, 224
torch.Size([1, 256, 56, 56])
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 2048, 7, 7])
torch.Size([1, 1000])

ref
https://note.youdao.com/ynoteshare1/index.html?id=5a7dbe1a71713c317062ddeedd97d98e&type=note
"""
import torch
from torch import nn


# ResNet Bottleneck_block_constructor
class Bottleneck_block_constructor(nn.Module):

    extention = 4

    # 定义初始化的网络和参数
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck_block_constructor, self).__init__()

        outplane = midplane * self.extention

        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)

        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)

        self.conv3 = nn.Conv2d(midplane, outplane, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midplane * self.extention)

        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual

        out = self.relu(out)

        return out


# Hybrid_backbone of ResNets
class ResNet_backbone(nn.Module):

    def __init__(self, block_constructor, bottleneck_channels_setting=None, identity_layers_setting=None,
                 stage_stride_setting=None, fc_num_classes=None, feature_idx=None, edge_size=224):

        if bottleneck_channels_setting is None:
            bottleneck_channels_setting = [64, 128, 256, 512]
        if identity_layers_setting is None:
            identity_layers_setting = [3, 4, 6, 3]
        if stage_stride_setting is None:
            stage_stride_setting = [1, 2, 2, 2]

        self.inplane = 64
        self.fc_num_classes = fc_num_classes
        self.feature_idx = feature_idx

        self.block_constructor = block_constructor  # Bottleneck_block_constructor
        self.num_features = 512 * self.block_constructor.extention

        super(ResNet_backbone, self).__init__()

        self.bcs = bottleneck_channels_setting  # [64, 128, 256, 512]
        self.ils = identity_layers_setting  # [3, 4, 6, 3]
        self.sss = stage_stride_setting  # [1, 2, 2, 2]

        # stem
        # alter the RGB pic chanel to match inplane
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # ResNet stages
        self.layer1 = self.make_stage_layer(self.block_constructor, self.bcs[0], self.ils[0], self.sss[0])
        self.layer2 = self.make_stage_layer(self.block_constructor, self.bcs[1], self.ils[1], self.sss[1])
        self.layer3 = self.make_stage_layer(self.block_constructor, self.bcs[2], self.ils[2], self.sss[2])
        self.layer4 = self.make_stage_layer(self.block_constructor, self.bcs[3], self.ils[3], self.sss[3])

        # last pool (not activate in backbone process)
        if edge_size == 224:
            self.avgpool = nn.AvgPool2d(7)
        elif edge_size == 384:
            self.avgpool = nn.AvgPool2d(12)
        else:
            print('not a avaliable edge size with ResNet backbone')
            raise

        # cls head
        if self.fc_num_classes is not None:
            self.fc = nn.Linear(self.num_features, fc_num_classes)

    def forward(self, x):

        # stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        stem_out = self.maxpool(out)

        # Resnet block of 4 stages
        stage1_out = self.layer1(stem_out)
        stage2_out = self.layer2(stage1_out)
        stage3_out = self.layer3(stage2_out)
        stage4_out = self.layer4(stage3_out)

        if self.fc_num_classes is not None:
            # connect to cls head mlp if asked
            fc_out = self.avgpool(stage4_out)
            fc_out = torch.flatten(fc_out, 1)
            fc_out = self.fc(fc_out)

        # get what we need for different usage
        if self.feature_idx == 'stages':
            if self.fc_num_classes is not None:
                return stage1_out, stage2_out, stage3_out, stage4_out, fc_out
            else:
                return stage1_out, stage2_out, stage3_out, stage4_out
        elif self.feature_idx == 'features':
            if self.fc_num_classes is not None:
                return stem_out, stage1_out, stage2_out, stage3_out, stage4_out, fc_out
            else:
                return stem_out, stage1_out, stage2_out, stage3_out, stage4_out
        else:  # self.feature_idx is None
            if self.fc_num_classes is not None:
                return fc_out
            else:
                return stage4_out

    def make_stage_layer(self, block_constractor, midplane, block_num, stride=1):
        """
        block:
        midplane：usually = output chanel/4
        block_num：
        stride：stride of the ResNet Conv Block
        """

        block_list = []

        outplane = midplane * block_constractor.extention  # extention

        if stride != 1 or self.inplane != outplane:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, outplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block_constractor.extention)
            )
        else:
            downsample = None

        # Conv Block
        conv_block = block_constractor(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)

        self.inplane = outplane  # update inplane for the next stage

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block_constractor(self.inplane, midplane, stride=1, downsample=None))

        return nn.Sequential(*block_list)  # stack blocks


def build_ResNet50_backbone(edge_size=224, pretrained=True):
    backbone = ResNet_backbone(block_constructor=Bottleneck_block_constructor,
                               bottleneck_channels_setting=[64, 128, 256, 512],
                               identity_layers_setting=[3, 4, 6, 3],
                               stage_stride_setting=[1, 2, 2, 2],
                               fc_num_classes=None,
                               feature_idx=None,
                               edge_size=edge_size)

    if pretrained:
        from torchvision import models
        backbone_weights = models.resnet50(pretrained=True).state_dict()
        # True for pretrained Resnet50 model, False will randomly initiate
    else:
        backbone_weights = None

    if pretrained:
        try:
            backbone.load_state_dict(backbone_weights, False)
        except:
            print("backbone loading erro!")
        else:
            print("backbone loaded")
    else:
        print("backbone not loaded")

    return backbone
