# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmyolo.registry import MODELS
from mmrazor.models.architectures.ops.base import BaseOP

from ..utils.misc import channel_shuffle

@MODELS.register_module()
class ShuffleBlock(BaseOP):
    """
        refer to mmrazor.models.architectures.ops.shufflenet_series
    """

    def __init__(self,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):

        super(ShuffleBlock, self).__init__(**kwargs)

        assert kernel_size in [3, 5, 7]
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        branch_features = self.out_channels // 2
        if self.stride == 1:
            assert self.in_channels == branch_features * 2, (
                f'in_channels ({self.in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if self.in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                    groups=self.in_channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None),
                ConvModule(
                    self.in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                self.in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                groups=branch_features,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out


@MODELS.register_module()
class ShuffleXception(BaseOP):
    """Xception block for ShuffleNetV2 backbone.
        refer to mmrazor.models.architectures.ops.shufflenet_series
    """

    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(ShuffleXception, self).__init__(**kwargs)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.mid_channels = self.out_channels // 2

        branch_features = self.out_channels // 2
        if self.stride == 1:
            assert self.in_channels == branch_features * 2, (
                f'in_channels ({self.in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if self.in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=self.in_channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None),
                ConvModule(
                    self.in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
            )

        self.branch2 = []

        self.branch2.append(
            DepthwiseSeparableConvModule(
                self.in_channels if (self.stride > 1) else branch_features,
                self.mid_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg), )
        self.branch2.append(
            DepthwiseSeparableConvModule(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg))
        self.branch2.append(
            DepthwiseSeparableConvModule(
                self.mid_channels,
                branch_features,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg))
        self.branch2 = nn.Sequential(*self.branch2)

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out
