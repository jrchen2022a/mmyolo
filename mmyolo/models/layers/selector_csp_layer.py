# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.csp_layer import DarknetBottleneck, CSPNeXtBlock, CSPLayer
from mmdet.models.layers.se_layer import ChannelAttention

import torch.nn.functional as F

from . import CSPLayerWithTwoConv


class AttentionSEblock(nn.Module):
    def __init__(self, channels, num_outs, reduction: int = 4):  # , temperature
        super(AttentionSEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, num_outs)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        # self.temperature = temperature
        self.channels = channels

    def forward(self, x):
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.gumbel_softmax(x, tau=1, hard=True)
        return x


class SelectorCSPLayer(BaseModule):
    """
        改动short_conv为selector，效果一般
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_selectors: int = 3,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

        self.short_conv_group = nn.ModuleList()
        for idx in range(num_selectors):
            self.short_conv_group.append(
                ConvModule(
                    in_channels,
                    mid_channels,
                    1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.switch = AttentionSEblock(in_channels, num_outs=num_selectors, reduction=4)

    def forward(self, x: Tensor) -> Tensor:
        switch_res = torch.split(self.switch(x), 1, dim=1)
        x_short = None
        for (gate, single_conv) in zip(switch_res, self.short_conv_group):
            if x_short is None:
                x_short = single_conv(x) * gate.view(x.shape[0], 1, 1, 1)
            else:
                x_short += single_conv(x) * gate.view(x.shape[0], 1, 1, 1)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SelectorCSPLayerV2(CSPLayer):
    """
        改动main_conv中的bottleneck为selector，效果好一点
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_selectors: int = 3,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
            use_depthwise=use_depthwise,
            use_cspnext_block=use_cspnext_block,
            channel_attention=channel_attention,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.selectors_group = nn.ModuleList(
            copy.deepcopy(self.blocks) for _ in range(num_selectors))
        del self.blocks
        self.switch = AttentionSEblock(in_channels, num_outs=num_selectors, reduction=4)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main_mid = self.main_conv(x)
        x_switch = self.switch(x)
        if self.training:
            switch_res = torch.split(x_switch, 1, dim=1)
            x_main = None
            for (gate, single_blocks) in zip(switch_res, self.selectors_group):
                if x_main is None:
                    x_main = single_blocks(x_main_mid) * gate.view(x.shape[0], 1, 1, 1)
                else:
                    x_main += single_blocks(x_main_mid) * gate.view(x.shape[0], 1, 1, 1)
        else:
            x_main = self.selectors_group[x_switch.squeeze().nonzero().squeeze()](x_main_mid)

        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

class SelectorCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """
        给v8的selector
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_selectors: int = 3,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.switch = AttentionSEblock(in_channels, num_outs=num_selectors, reduction=4)
        self.selectors_group = nn.ModuleList(
            self.blocks for _ in range(num_selectors))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_switch = self.switch(x)

        x_main = self.main_conv(x)

        # for (gate, single_blocks) in zip(x_switch, self.blocks_group):
        #     if x_main is None:
        #         x_main = single_blocks(x_main_mid) * gate.view(x.shape[0], 1, 1, 1)
        #     else:
        #         x_main += single_blocks(x_main_mid) * gate.view(x.shape[0], 1, 1, 1)
        #
        # x_final = torch.cat((x_main, x_short), dim=1)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))
