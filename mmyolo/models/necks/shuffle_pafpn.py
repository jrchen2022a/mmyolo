from typing import List, Union

import torch.nn as nn
from mmdet.utils import ConfigType

from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov8_pafpn import YOLOv8PAFPN
from ..layers import build_shuffle_series


@MODELS.register_module()
class SufflePAFPN(YOLOv8PAFPN):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 shuffle_arch_setting: List,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        assert len(shuffle_arch_setting) == 2
        self.num_shuffle_blocks, self.shuffle_block_cfg = shuffle_arch_setting
        self.shuffle_block_cfg.update(norm_cfg=norm_cfg, act_cfg=act_cfg)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        return nn.Upsample(scale_factor=0.5, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        in_channels = make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                       self.widen_factor)
        out_channels = make_divisible(self.out_channels[idx - 1], self.widen_factor)
        return build_shuffle_series(
                    block_cfg=self.shuffle_block_cfg,
                    num_blocks=self.num_shuffle_blocks,
                    in_channels=in_channels,
                    out_channels=out_channels)


    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        in_channels = make_divisible((self.in_channels[idx] + self.in_channels[idx + 1]),
                                     self.widen_factor)
        out_channels = make_divisible(self.out_channels[idx + 1], self.widen_factor)
        return build_shuffle_series(
                    block_cfg=self.shuffle_block_cfg,
                    num_blocks=self.num_shuffle_blocks,
                    in_channels=in_channels,
                    out_channels=out_channels)

