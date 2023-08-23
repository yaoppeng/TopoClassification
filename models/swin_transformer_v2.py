import os

from models.swin_utils import *
from models.pH.pers_lay import *
from models.pH.pllay import *
from models.pointnet.pointnet_utils import PointNetEncoder
from models.swin_utils import *
import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "SwinTransformer",
    "Swin_T_Weights",
    "Swin_S_Weights",
    "Swin_B_Weights",
    "Swin_V2_T_Weights",
    "Swin_V2_S_Weights",
    "Swin_V2_B_Weights",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))

        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # print(self)
        self.load_state_dict(torch.load(join(os.getenv("HOME"), ".cache/torch/hub/checkpoints/"
                                        "swin_v2_b-781e5279.pth")))
        self.update()

    def update(self):
        self.patch_embedding = self.features[0]
        # print(self.patch_embedding[0].weight)
        self.stage_1 = self.features[1]
        self.stage_2 = nn.Sequential(*self.features[2:4])

        self.stage_3 = nn.Sequential(*self.features[4:6])
        self.stage_4 = nn.Sequential(*self.features[6:8])

        print(f"delete features =================================")
        del self.features

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x, pd, pl = x

        x = self.patch_embedding(x)  # (3, 224, 224) -> (56, 56, 128)
        x = self.stage_1(x)  # (56, 56, 128) -> (56, 56, 128)
        x = self.stage_2(x)  # (56, 56, 128) -> (28, 28, 256)
        x = self.stage_3(x)  # (28, 28, 256) -> (14, 14, 512)
        x = self.stage_4(x)  # (14, 14, 512) -> (7, 7, 1024)

        # x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)  # (7, 7, 1024) -> (1024, 7, 7)
        x = self.avgpool(x)  # (1024, 7, 7) -> (1024, 1, 1)
        x = self.flatten(x)  # (1024, 1, 1) -> (1024)
        x = self.head(x)  # (1024) -> (2)
        return x, torch.tensor(0)


class TopoSwinTransformer(SwinTransformer):
    def __init__(self,
                 patch_size: List[int],
                 embed_dim: int,
                 depths: List[int],
                 num_heads: List[int],
                 window_size: List[int],
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 stochastic_depth_prob: float = 0.1,
                 num_classes: int = 1000,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 block: Optional[Callable[..., nn.Module]] = None,
                 downsample_layer: Callable[..., nn.Module] = PatchMerging,
                 topo_setting: int = 0,
                 share_topo: bool = True,
                 topo_layers: List = None):
        self.topo_setting = topo_setting
        self.share_topo = share_topo
        self.topo_layers = topo_layers
        super().__init__(
            patch_size,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            dropout,
            attention_dropout,
            stochastic_depth_prob,
            num_classes,
            norm_layer,
            block,
            downsample_layer,
        )
        self.topo_linear = nn.Linear(2048, num_classes)

    def update(self):
        self.pointnet_encoder = PointNetEncoder(global_feat=True, channel=4, feature_transform=False)
        self.linear_1 = nn.Sequential(nn.Linear(2048, 128))
        self.linear_2 = nn.Sequential(nn.Linear(2048, 256))
        self.linear_3 = nn.Sequential(nn.Linear(2048, 512))
        self.linear_4 = nn.Sequential(nn.Linear(2048, 1024))

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

        self.patch_embedding = self.features[0]
        # print(self.patch_embedding[0].weight)
        self.stage_1 = self.features[1]
        self.stage_2 = nn.Sequential(*self.features[2:4])
        self.stage_3 = nn.Sequential(*self.features[4:6])
        self.stage_4 = nn.Sequential(*self.features[6:8])

        print(f"delete features =================================")
        del self.features

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x, pd, pl = x
        else:
            raise ValueError(f"invalid input: {x}")
        x = self.patch_embedding(x)  # (3, 224, 224) -> (56, 56, 128)
        x = self.stage_1(x)  # (56, 56, 128) -> (56, 56, 128)

        if self.share_topo:
            o, trans, trans_feat = self.topology_branch(pd)
        else:
            o, trans, trans_feat = self.topology_branch[0](pd)

        out_topo = o
        if self.topo_layers[0]:
            # first topology
            o1 = self.linear_1(o)
            scale1 = F.sigmoid(o1).unsqueeze(1).unsqueeze(2).expand_as(x)
            out1 = x * scale1
            x = x + out1

        # second topology
        x = self.stage_2(x)  # (56, 56, 128) -> (28, 28, 256)
        if not self.share_topo:
            o, trans, trans_feat = self.topology_branch[1](pd)

        if self.topo_layers[1]:
            o2 = self.linear_2(o)
            scale2 = F.sigmoid(o2).unsqueeze(1).unsqueeze(2).expand_as(x)
            out2 = x * scale2
            x = x + out2

        # third topology
        x = self.stage_3(x)  # (28, 28, 256) -> (14, 14, 512)
        if not self.share_topo:
            o, trans, trans_feat = self.topology_branch[2](pd)

        if self.topo_layers[2]:
            o3 = self.linear_3(o)
            scale3 = F.sigmoid(o3).unsqueeze(1).unsqueeze(2).expand_as(x)
            out3 = x * scale3
            x = x + out3

        # fourth topology
        x = self.stage_4(x)  # (14, 14, 512) -> (7, 7, 1024)
        if not self.share_topo:
            o, trans, trans_feat = self.topology_branch[3](pd)
        if self.topo_layers[3]:
            o4 = self.linear_4(o)
            scale4 = F.sigmoid(o4).unsqueeze(1).unsqueeze(2).expand_as(x)
            out4 = x * scale4
            x = x + out4

        # x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)  # (7, 7, 1024) -> (1024, 7, 7)
        x = self.avgpool(x)  # (1024, 7, 7) -> (1024, 1, 1)
        x = self.flatten(x)  # (1024, 1, 1) -> (1024)
        x = self.head(x)  # (1024) -> (2)

        return x, self.topo_linear(out_topo)


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
}
# @register_model()
# @handle_legacy_interface(weights=("pretrained", Swin_B_Weights.IMAGENET1K_V1))
def swin_b(*, weights: Optional[Swin_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_B_Weights
        :members:
    """
    weights = Swin_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        **kwargs,
    )

# @register_model()
# @handle_legacy_interface(weights=("pretrained", Swin_V2_B_Weights.IMAGENET1K_V1))
def swin_v2_b(*, weights: Optional[Swin_V2_B_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    weights = Swin_V2_B_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
