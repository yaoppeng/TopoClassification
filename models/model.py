"""Model.

    Model should be formulated here.
"""
from models.resnet.resnet50 import resnet50, ResNet152
from models.senet.senet import senet154
import sys
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

import pretrainedmodels


# from utils.initialization import _kaiming_normal, _xavier_normal, \
#         _kaiming_uniform, _xavier_uniform


class Network(nn.Module):
    """Network
    """

    def __init__(self, backbone="resnet50", num_classes=7, input_channel=3,
                 pretrained=True, use_topology=False, topo_layers=None, share_topo=True,
                 he_init=True, use_cnn=True, topo_model="", concate=False,
                 topo_setting=0):
        super(Network, self).__init__()
        if backbone == "resnet50":
            model = ResNet50(num_classes=num_classes,
                             input_channel=input_channel,
                             pretrained=pretrained,
                             use_topology=use_topology,
                             topo_layers=topo_layers,
                             share_topo=share_topo,
                             he_init=he_init,
                             topo_model=topo_model,
                             topo_setting=topo_setting)
        elif backbone == "resnet152":
            model = ResNet152(num_classes=num_classes,
                              input_channel=input_channel,
                              pretrained=pretrained,
                              use_topology=use_topology,
                              topo_layers=topo_layers,
                              share_topo=share_topo,
                              he_init=he_init, use_cnn=use_cnn,
                              topo_model=topo_model,
                              concate=concate,
                              topo_setting=topo_setting)
        elif backbone == "resnet18":
            model = ResNet18(num_classes=num_classes,
                             input_channel=input_channel,
                             pretrained=pretrained,
                             topo_setting=topo_setting)
        elif backbone == "senet154":
            model = SENet154(num_classes=num_classes,
                             pretrained=pretrained,
                             use_topology=use_topology,
                             topo_layers=topo_layers,
                             share_topo=share_topo,
                             he_init=he_init,
                             use_cnn=use_cnn,
                             topo_model=topo_model,
                             concate=concate,
                             topo_setting=topo_setting)
        elif backbone == "PNASNet5Large":
            model = PNASNet5Large(num_classes=num_classes,
                                  input_channel=input_channel,
                                  pretrained=pretrained)
        elif backbone == "NASNetALarge":
            model = NASNetALarge(num_classes=num_classes,
                                 input_channel=input_channel,
                                 pretrained=pretrained)
        else:
            print("Need model")
            sys.exit(-1)

        self.model = model
        self.topo_linear = nn.Linear(2048, num_classes)

    def forward(self, inputs):
        predict, topo = self.model(inputs[0], inputs[1], inputs[2])
        topo = self.topo_linear(topo)
        return predict, topo

    def print_model(self, input_size, device):
        """Print models structure
        """
        self.model.to(device)
        summary(self.model, input_size)


class ResNet50(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained, use_topology=False, topo_layers=None, share_topo=True,
                 he_init=True, only_topo=True, topo_model="",
                 topo_setting=0):
        super(ResNet50, self).__init__()
        # self.features = nn.Sequential(
        #     *list(torchvision.models.resnet50(pretrained=pretrained).
        #           children())[:-1]
        #     )
        self.topo_setting = topo_setting
        self.model = resnet50(pretrained=pretrained, use_topology=use_topology, topo_layers=topo_layers,
                             share_topo=share_topo, he_init=he_init, only_topo=only_topo, topo_model=topo_model)
        # self.features = nn.Sequential(
        #     *list(resnet50(pretrained=pretrained, use_topology=use_topology).
        #           children())[:-1]
        # )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, x1, x2):
        # x = self.features([x, x1])
        x = self.model(x, x1, x2)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(ResNet18, self).__init__()
        self.features = nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=pretrained).
                  children())[:-1]
            )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PNASNet5Large(nn.Module):
    """PNASNet5Large.
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(PNASNet5Large, self).__init__()
        model = pretrainedmodels.pnasnet5large(num_classes=1000,
                                               pretrained="imagenet")
        model.last_linear = nn.Linear(model.last_linear.in_features,
                                      num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


class NASNetALarge(nn.Module):
    """NASNetALarge.
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(NASNetALarge, self).__init__()
        model = pretrainedmodels.nasnetalarge(num_classes=1000,
                                              pretrained="imagenet")
        model.last_linear = nn.Linear(model.last_linear.in_features,
                                      num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


class Identity(nn.Module):
    """Identity path.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


# def init_weights(net, method, _print):
#    """Initialize weights of the networks.
#
#        weights of conv layers and fully connected layers are both initialzied
#        with Xavier algorithm. In particular, set parameters to random values
#        uniformly drawn from [-a, a], where a = sqrt(6 * (din + dout)), for
#        batch normalization layers, y=1, b=0, all biases initialized to 0
#    """
#    if method == "kaiming_normal":
#        net = _kaiming_normal(net, _print)
#    elif method == "kaiming_uniform":
#        net = _kaiming_uniform(net, _print)
#    elif method == "xavier_uniform":
#        net = _xavier_uniform(net, _print)
#    elif method == "xavier_normal":
#        net = _xavier_normal(net, _print)
#    else:
#        _print("Init weight: Need legal initialization method")
#    return net

class SENet154(nn.Module):
    def __init__(self, num_classes, pretrained, use_topology=False, topo_layers=None, share_topo=True,
                 he_init=True, use_cnn=True, topo_model="", concate=False,
                 topo_setting=0):
        super(SENet154, self).__init__()

        self.model = senet154(pretrained=pretrained, use_topology=use_topology, topo_layers=topo_layers,
                             share_topo=share_topo, he_init=he_init, use_cnn=use_cnn, topo_model=topo_model,
                             concate=concate, topo_setting=topo_setting)

        if concate:
            self.classifier = nn.Linear(2048*2, num_classes)
        else:
            self.classifier = nn.Linear(2048, num_classes)

    def forward(self,x, pd, pl):
        # x = self.features([x, x1])
        x, topo_feature = self.model(x, pd, pl)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, topo_feature