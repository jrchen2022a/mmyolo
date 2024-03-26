from typing import List, Union

import torch.nn as nn
from mmdet.utils import ConfigType

from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov8_pafpn import YOLOv8PAFPN
from ..layers import build_lite_series
from mmcv.cnn import DepthwiseSeparableConvModule


@MODELS.register_module()
class LitePAFPN(YOLOv8PAFPN):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 lite_arch_setting: List,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.lite_block_cfg = lite_arch_setting
        self.lite_block_cfg.update(norm_cfg=norm_cfg, act_cfg=act_cfg)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

    # def build_downsample_layer(self, idx: int) -> nn.Module:
    #     return nn.Upsample(scale_factor=0.5, mode='nearest')

    def build_top_down_layer(self, idx: int) -> nn.Module:
        in_channels = make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                       self.widen_factor)
        out_channels = make_divisible(self.out_channels[idx - 1], self.widen_factor)
        return build_lite_series(
                    block_cfg=self.lite_block_cfg,
                    in_channels=in_channels,
                    out_channels=out_channels)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        in_channels = make_divisible((self.in_channels[idx] + self.in_channels[idx + 1]),
                                     self.widen_factor)
        out_channels = make_divisible(self.out_channels[idx + 1], self.widen_factor)
        return build_lite_series(
                    block_cfg=self.lite_block_cfg,
                    in_channels=in_channels,
                    out_channels=out_channels)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        return DepthwiseSeparableConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
