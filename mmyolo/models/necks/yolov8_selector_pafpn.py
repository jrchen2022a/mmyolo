# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..utils import make_divisible, make_round
from .yolov8_pafpn import YOLOv8PAFPN
from ..backbones.selector_csp import SELECTOR_SETTINGS


@MODELS.register_module()
class YOLOv8SelectorPAFPN(YOLOv8PAFPN):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 num_selectors: int = 3,
                 selector_hard: bool = False,
                 selector_type: str = 'SelectorCSPLayerWithTwoConv',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.num_selectors = num_selectors
        self.selector = SELECTOR_SETTINGS[selector_type]
        self.selector_hard = selector_hard
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        return self.selector(
                make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                               self.widen_factor),
                make_divisible(self.out_channels[idx - 1], self.widen_factor),
                num_selectors=self.num_selectors,
                selector_hard=self.selector_hard,
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return self.selector(
                make_divisible((self.in_channels[idx] + self.in_channels[idx + 1]),
                               self.widen_factor),
                make_divisible(self.out_channels[idx + 1], self.widen_factor),
                num_selectors=self.num_selectors,
                selector_hard=self.selector_hard,
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
