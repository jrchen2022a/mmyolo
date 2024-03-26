import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from mmrazor.models.architectures.ops.base import BaseOP

from ..utils.misc import channel_shuffle


def build_lite_series(block_cfg, in_channels, out_channels) -> nn.Module:
    block_cfg.update(
        in_channels=in_channels,
        out_channels=out_channels)
    return MODELS.build(block_cfg)


def build_shuffle_series(block_cfg, num_blocks, in_channels, out_channels) -> nn.Module:
    shuffle_stage = []
    if in_channels != out_channels:
        shuffle_stage.append(
            ConvModule(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=1,
                       norm_cfg=block_cfg.norm_cfg,
                       act_cfg=block_cfg.act_cfg))
    for i in range(num_blocks):
        block_cfg.update(
            in_channels=out_channels,
            out_channels=out_channels)
        shuffle_stage.append(MODELS.build(block_cfg))
    return nn.Sequential(*shuffle_stage)

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
class LiteBlock(BaseModule):
    """
        refer to paper
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_block:int=1,
                 conv_groups=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):

        super(LiteBlock, self).__init__(**kwargs)

        self.in_conv = None \
            if in_channels == out_channels \
            else ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)

        branch_features = out_channels // 2
        self.branch2 = nn.Sequential(*[
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=conv_groups,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_block)])

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1) if self.in_conv is None else self.in_conv(x).chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        # out = self.out_conv(channel_shuffle(out, 2))
        out = channel_shuffle(out, 2)
        return out

