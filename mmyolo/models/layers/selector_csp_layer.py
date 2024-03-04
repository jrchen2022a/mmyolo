# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmyolo.models.layers.yolo_bricks import DarknetBottleneck as V8DarknetBottleneck
from torch import Tensor

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.csp_layer import DarknetBottleneck, CSPNeXtBlock, CSPLayer

import torch.nn.functional as F
from . import CSPLayerWithTwoConv


class AttentionSEblock(nn.Module):
    def __init__(self, channels, num_outs, reduction: int = 4, hard: bool = False):  # , temperature
        super(AttentionSEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, num_outs)
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        # self.temperature = temperature
        self.channels = channels
        self.hard = hard

    def forward(self, x):
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.gumbel_softmax(x, tau=1, hard=True) if self.hard else F.softmax(x, dim=1)
        return x


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
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        del self.blocks
        self.selectors_group = nn.ModuleList(
            nn.Sequential(*[
                block(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ]) for _ in range(num_selectors))
        self.switch = AttentionSEblock(in_channels, num_outs=num_selectors, reduction=4)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main_mid = self.main_conv(x)
        x_switch = self.switch(x)
        switch_res = torch.split(x_switch, 1, dim=1)
        x_main = None
        for (gate, single_blocks) in zip(switch_res, self.selectors_group):
            if x_main is None:
                x_main = single_blocks(x_main_mid) * gate.view(x.shape[0], 1, 1, 1)
            else:
                x_main += single_blocks(x_main_mid) * gate.view(x.shape[0], 1, 1, 1)

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
            selector_hard: bool = False,
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
        self.num_blocks = num_blocks
        del self.blocks
        self.selectors_group = nn.ModuleList(
            nn.ModuleList(
                V8DarknetBottleneck(
                    self.mid_channels,
                    self.mid_channels,
                    expansion=1,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    add_identity=add_identity,
                    use_depthwise=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)) for _ in range(num_selectors))
        self.switch = AttentionSEblock(in_channels, num_outs=num_selectors, reduction=4, hard=selector_hard)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_switch = self.switch(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        switch_res = torch.split(x_switch, 1, dim=1)
        x_extend = list()
        for idx, (gate, single_blocks) in enumerate(zip(switch_res, self.selectors_group)):
            sx_in = x_main[1]
            for (i, blocks) in enumerate(single_blocks):
                sx_int = blocks(sx_in) * gate.view(x.shape[0], 1, 1, 1)
                if len(x_extend) is i:
                    x_extend.append(sx_int)
                else:
                    x_extend[i] = x_extend[i] + sx_int
                sx_in = sx_int
        x_main.extend(x_extend)

        return self.final_conv(torch.cat(x_main, 1))
