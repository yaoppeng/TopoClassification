"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math
import torch

from models.pH.pers_lay import *
from models.pH.pllay import *
from models.pointnet.pointnet_utils import PointNetEncoder

import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class PointNetFC(nn.Module):
    def __init__(self, channels, reduction):
        super(PointNetFC, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, xlists):
        if isinstance(xlists, (list, tuple)):
            x, x1, x2 = xlists[0], xlists[1], xlists[2]
        else:
            x = xlists
            x1 = None
            x2 = None

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # add topology branch at each block

        out = self.se_module(out)

        # o = None
        # if self.topology_branch is not None and (x1 is not None or x2 is not None):
        #     # o, trans, trans_feat = self.topology_branch(x1)
        #
        #     if isinstance(self.topology_branch, PointNetEncoder):
        #         o, trans, trans_feat = self.topology_branch(x1)
        #     else:
        #         o = self.topology_branch(x2)
        #
        #     scale = self.pointnet_fc(o)
        #     # scale = F.sigmoid(o).unsqueeze(2).unsqueeze(3).expand_as(out)
        #
        #     out *= scale

        out = out + residual
        # out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, pointnet=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

        if pointnet is not None:
            self.topology_branch = pointnet
            self.pointnet_fc = PointNetFC(planes * self.expansion, reduction=16)
            # self.pointnet_fc = nn.Linear(2048, planes * self.expansion)
        else:
            self.topology_branch = None


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, pointnet=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

        if pointnet is not None:
            self.topology_branch = pointnet
            self.pointnet_fc = PointNetFC(planes * self.expansion, reduction=16)
            # self.pointnet_fc = nn.Linear(2048, planes * self.expansion)
        else:
            self.topology_branch = None


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1,he_init=True,
                 use_cnn=True, share_topo=True, use_topology: bool = True,
                 topo_layers=None,  topo_model="", concate=False, topo_setting=0):
        super(SENet, self).__init__()
        self.topo_setting = topo_setting
        self.share_topo = share_topo
        self.topo_model = topo_model
        self.use_cnn = use_cnn
        self.concate = concate

        if not use_cnn or concate:
            if self.topo_model == "perslay":
                # using persistent landscape
                self.pers_lay = PersLay()
            elif self.topo_model == "pllay":
                # using persistent landscape
                self.pllay = PLLay()
            elif self.topo_model == "pointnet":
                # using persistent diagram
                self.pointnet = PointNetEncoder(global_feat=True, channel=4)
            else:
                raise ValueError(f"invalide topo model: {self.topo_model}")

            if not use_cnn:
                return

        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        if self.topo_setting == 0:
            if self.share_topo:
                self.topology_branch = PointNetEncoder(global_feat=True, channel=4)
            else:
                self.topology_branch = nn.ModuleList([
                    PointNetEncoder(global_feat=True, channel=4) for i in range(4)
                ])
        elif self.topo_setting == 1:
            self.topology_branch = PersLay()
        elif self.topo_setting == 2:
            self.topology_branch = PLLay()
        else:
            raise ValueError(f"invalid topo setting: {self.topo_setting}")

        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        if share_topo:
            if self.topo_model == "perslay":
                topology = PersLay()
            elif self.topo_model == "pllay":
                topology = PLLay() if use_topology else None
            elif self.topo_model == "pointnet":
                topology = PointNetEncoder(global_feat=True, feature_transform=False, channel=4, he_init=he_init) \
                    if use_topology else None
            else:
                raise ValueError(f"invalid model {self.topo_model}")

            if topo_layers is not None and topo_layers[3]:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               groups=groups,
                                               reduction=reduction,
                                               downsample_kernel_size=downsample_kernel_size,
                                               downsample_padding=downsample_padding,
                                               pointnet=topology)
        else:
            if self.topo_model == "perslay":
                topology = PersLay()
            elif self.topo_model == "pllay":
                topology = PLLay() if use_topology else None
            elif self.topo_model == "pointnet":
                topology = PointNetEncoder(global_feat=True, feature_transform=False,
                                           channel=4, he_init=he_init) \
                    if use_topology else None
            else:
                raise ValueError(f"invalid model {self.topo_model}")

            if topo_layers is not None and topo_layers[3]:
                self.layer4 = self._make_layer(
                    block,
                    planes=512,
                    blocks=layers[3],
                    stride=2,
                    groups=groups,
                    reduction=reduction,
                    downsample_kernel_size=downsample_kernel_size,
                    downsample_padding=downsample_padding,
                    pointnet=topology)
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.topo_layers = topo_layers
        self.linear_1 = nn.Sequential(nn.Linear(2048, 256))
        self.linear_2 = nn.Sequential(nn.Linear(2048, 512))
        self.linear_3 = nn.Sequential(nn.Linear(2048, 1024))
        self.linear_4 = nn.Sequential(nn.Linear(2048, 2048))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0, pointnet=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample, pointnet))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction, pointnet=pointnet))

        return nn.Sequential(*layers)

    def features(self, x, pd, pl):

        if not self.use_cnn:
            if self.topo_model == 'perslay':
                return None, self.pers_lay(pl)
            elif self.topo_model == 'pllay':
                return None, self.pllay(pl)
            elif self.topo_model == 'pointnet':
                return None, self.pointnet(pd)
            else:
                raise ValueError(f"invalid topo_model: {self.topo_model}")

        x = self.layer0(x)

        if self.concate:
            x, pd_1, pl_1, out_feature = self.layer1([x, None, None])
            x, pd_1, pl_1, out_feature = self.layer2([x, None, None])
            x, pd_1, pl_1, out_feature = self.layer3([x, None, None])
            x, feature, pl_1, out_feature = self.layer4([x, None, None])

            if self.topo_model == 'perslay':
                f = self.pers_lay(pl)
            elif self.topo_model == 'pllay':
                f = self.pllay(pl)
            elif self.topo_model == 'pointnet':
                f, _, _ = self.pointnet(pd)
            else:
                raise ValueError(f"invalid topo_model: {self.topo_model}")

            return x, f

        if self.share_topo:
            o, trans, trans_feat = self.topology_branch(pd)
        else:
            o, trans, trans_feat = self.topology_branch[0](pd)

        out_topo = o
        x = self.layer1(x)
        if self.topo_layers[0]:
            # first topology
            o1 = self.linear_1(o)
            scale1 = F.sigmoid(o1).unsqueeze(2).unsqueeze(3).expand_as(x)
            out1 = x * scale1
            x = x + out1

        x = self.layer2(x)
        if not self.share_topo:
            o, trans, trans_feat = self.topology_branch[1](pd)
        if self.topo_layers[1]:
            o2 = self.linear_2(o)
            scale2 = F.sigmoid(o2).unsqueeze(2).unsqueeze(3).expand_as(x)
            out2 = x * scale2
            x = x + out2

        x = self.layer3(x)
        if not self.share_topo:
            o, trans, trans_feat = self.topology_branch[2](pd)

        if self.topo_layers[2]:
            o3 = self.linear_3(o)
            scale3 = F.sigmoid(o3).unsqueeze(2).unsqueeze(3).expand_as(x)
            out3 = x * scale3
            x = x + out3

        x = self.layer4(x)  # (14, 14, 512) -> (7, 7, 1024)
        if not self.share_topo:
            o, trans, trans_feat = self.topology_branch[3](pd)
        if self.topo_layers[3]:
            o4 = self.linear_4(o)
            scale4 = F.sigmoid(o4).unsqueeze(2).unsqueeze(3).expand_as(x)
            out4 = x * scale4
            x = x + out4

        # x, pd, pl, out_feature = self.layer1([x, pd, pl])
        # x, pd, pl, out_feature = self.layer2([x, pd, pl])
        # x, pd, pl, out_feature = self.layer3([x, pd, pl])
        # x, pd, pl, out_feature = self.layer4([x, pd, pl])
        return x, out_topo

    def logits(self, x):
        if self.concate:
            x, f = x[0], x[1]
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)

        if self.concate:
            x = torch.cat([x, f], dim=-1)
        # x = self.last_linear(x)
        return x

    def forward(self, x, pd, pl):
        x, out_feature = self.features(x, pd, pl)

        if self.concate:
            x = self.logits([x, out_feature])
        else:
            x = self.logits(x)
        return x, out_feature


def initialize_pretrained_model(model, num_classes, settings):
    # assert num_classes == settings['num_classes'], \
    #     'num_classes should be {}, but is {}'.format(
    #         settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet', **kwargs):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['senet154']['imagenet']
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
