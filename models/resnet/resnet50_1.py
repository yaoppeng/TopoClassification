import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from models.pH.pers_lay import *
from models.pH.pllay import *
from models.pointnet.pointnet_utils import PointNetEncoder
import torch.nn.functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, pointnet=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if pointnet is not None:
            self.topology_branch = pointnet
            self.pointnet_fc = nn.Linear(2048, planes * self.expansion)
        else:
            self.topology_branch = None

    def forward(self, xlists):

        if isinstance(xlists, (list, tuple)):
            x, x1 = xlists[0], xlists[1]
        else:
            x = xlists
            x1 = None

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # add topology branch at each block
        topo_out = None
        if self.topology_branch is not None:
            o, trans, trans_feat = self.topology_branch(x1)
            topo_out = o
            o = self.pointnet_fc(o)
            scale = F.sigmoid(o).unsqueeze(2).unsqueeze(3).expand_as(out)

            out *= scale

        out += identity
        out = self.relu(out)

        if self.topology_branch is not None:
            return out, topo_out
        else:
            return out, topo_out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_topology: bool = True, topo_layers=None,
                 share_topo=True, he_init=True, only_topo=True, topo_model="",
                 use_cnn=True, concate=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.only_topo = not use_cnn
        self.topo_model = topo_model

        if use_topology:
            self.pers_lay = PersLay()
            self.pllay = PLLay()
            self.pointnet = PointNetEncoder(global_feat=True, channel=4)

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if share_topo:
            if self.topo_model == "perslay":
                topology = PLLay()
            elif self.topo_model == "pllay":
                topology = PersLay() if use_topology else None
            elif self.topo_model == "pointnet":
                topology = PointNetEncoder(global_feat=True, feature_transform=False, channel=4, he_init=he_init) \
                    if use_topology else None
            else:
                raise ValueError(f"invalid models {self.topo_model}")

            if topo_layers is not None and topo_layers[0]:
                self.layer1 = self._make_layer(block, 64, layers[0], pointnet=topology)
            else:
                self.layer1 = self._make_layer(block, 64, layers[0])

            if topo_layers is not None and topo_layers[1]:
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[0], pointnet=topology)
            else:
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[0])

            if topo_layers is not None and topo_layers[2]:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[1], pointnet=topology)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[1])

            if topo_layers is not None and topo_layers[3]:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               dilate=replace_stride_with_dilation[2], pointnet=topology)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               dilate=replace_stride_with_dilation[2])
        else:
            if use_topology and topo_layers is not None and topo_layers[0]:
                if self.topo_model == 'perslay':
                    topology_1 = PersLay()
                elif self.topo_model == 'pllay':
                    topology_1 = PLLay()
                elif self.topo_model == 'pointnet':
                    topology_1 = PointNetEncoder(global_feat=True,
                                                 feature_transform=False, channel=4, he_init=he_init)
                else:
                    raise ValueError(f"invalid topo models: {self.topo_model}")

                self.layer1 = self._make_layer(block, 64, layers[0], pointnet=topology_1)
            else:
                self.layer1 = self._make_layer(block, 64, layers[0])

            if use_topology and topo_layers is not None and topo_layers[1]:
                if self.topo_model == 'perslay':
                    topology_2 = PersLay()
                elif self.topo_model == 'pllay':
                    topology_2 = PLLay()
                elif self.topo_model == 'pointnet':
                    topology_2 = PointNetEncoder(global_feat=True,
                                                 feature_transform=False, channel=4, he_init=he_init)
                else:
                    raise ValueError(f"invalid topo models: {self.topo_model}")
                # topology_2 = PointNetEncoder(global_feat=True, feature_transform=False, channel=4, he_init=he_init)
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[0], pointnet=topology_2)
            else:
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[0])

            if use_topology and topo_layers is not None and topo_layers[2]:
                if self.topo_model == 'perslay':
                    topology_3 = PersLay()
                elif self.topo_model == 'pllay':
                    topology_3 = PLLay()
                elif self.topo_model == 'pointnet':
                    topology_3 = PointNetEncoder(global_feat=True,
                                                 feature_transform=False, channel=4, he_init=he_init)
                else:
                    raise ValueError(f"invalid topo models: {self.topo_model}")
                # topology_3 = PointNetEncoder(global_feat=True, feature_transform=False, channel=4, he_init=he_init)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[1], pointnet=topology_3)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[1])

            if use_topology and topo_layers is not None and topo_layers[3]:
                if self.topo_model == 'perslay':
                    topology_4 = PersLay()
                elif self.topo_model == 'pllay':
                    topology_4 = PLLay()
                elif self.topo_model == 'pointnet':
                    topology_4 = PointNetEncoder(global_feat=True,
                                                 feature_transform=False, channel=4, he_init=he_init)
                else:
                    raise ValueError(f"invalid topo models: {self.topo_model}")
                # topology_4 = PointNetEncoder(global_feat=True, feature_transform=False, channel=4, he_init=he_init)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               dilate=replace_stride_with_dilation[2], pointnet=topology_4)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the models by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, pointnet=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, pointnet=pointnet))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, pointnet=pointnet))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x1, x2):
        # See note [TorchScript super()]
        # x, x1 = xlists[0], xlists[1]

        if self.only_topo:
            if self.topo_model == "perslay":
                # using persistent landscape
                return torch.tensor(0).cuda(), self.pers_lay(x2)
            elif self.topo_model == "pllay":
                # using persistent landscape
                return torch.tensor(0), self.pllay(x2)
            elif self.topo_model == "pointnet":
                # using persistent diagram
                x, trans, trans_feat = self.pointnet(x1)
                return torch.tensor(0), x
            else:
                raise ValueError(f"invalide topo models: {self.topo_model}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.topo_model in ['perslay', 'pllay']:
            x, topo_feature = self.layer1([x, x2])
            # x = self.layer1([x, x1])  # the output is a lis now

            x, topo_feature = self.layer2([x, x2])

            x, topo_feature = self.layer3([x, x2])

            # x = self.layer4(x)
            x, topo_feature = self.layer4([x, x2])

        elif self.topo_model == 'pointnet':
            x, topo_feature = self.layer1([x, x1])
            # x = self.layer1([x, x1])  # the output is a lis now

            x, topo_feature = self.layer2([x, x1])

            x, topo_feature = self.layer3([x, x1])

            # x = self.layer4(x)
            x, topo_feature = self.layer4([x, x1])

        # x = self.layer4([x, x1])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x, topo_feature

    def forward(self, x, x1, x2):
        return self._forward_impl(x, x1, x2)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

        """
        nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        """

        # pretrained_weights = model.conv1.weight.data
        #
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        # model.conv1.weight.data = pretrained_weights[:, :1, :, :]

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d models from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d models from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 models from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The models is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 models from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The models is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class ResNet152(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained, use_topology=False, topo_layers=None, share_topo=True,
                 he_init=True, use_cnn=True, topo_model="", concate=False):
        super(ResNet152, self).__init__()
        # self.features = nn.Sequential(
        #     *list(torchvision.models.resnet50(pretrained=pretrained).
        #           children())[:-1]
        #     )
        self.use_cnn = use_cnn
        self.model = resnet152(pretrained=pretrained, use_topology=use_topology, topo_layers=topo_layers,
                             share_topo=share_topo, he_init=he_init, use_cnn=use_cnn,
                              topo_model=topo_model, concate=concate)
        # self.features = nn.Sequential(
        #     *list(resnet50(pretrained=pretrained, use_topology=use_topology).
        #           children())[:-1]
        # )
        if concate:
            self.classifier = nn.Linear(2048 * 2, num_classes)
        else:
            self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, x1, x2):
        # x = self.features([x, x1])
        x, topo_feature = self.model(x, x1, x2)
        if isinstance(x, (tuple, list)):
            x = x[0]

        if not self.use_cnn:
            topo_feature = self.classifier(topo_feature)
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x, topo_feature