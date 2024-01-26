# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .yolov8_pafpn import YOLOv8PAFPN
from ..plugins import ChannelAttention
from ..utils import make_divisible



@MODELS.register_module()
class YOLOv8FSPAFPN(YOLOv8PAFPN):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
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

    def build_out_layer(self, idx: int):
        """build reduce layer."""
        return ChannelAttention(
            make_divisible(self.out_channels[idx], self.widen_factor)
        )