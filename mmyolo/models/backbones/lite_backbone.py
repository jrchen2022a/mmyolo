from typing import Sequence, List, Union

from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.cnn import DepthwiseSeparableConvModule
from mmyolo.registry import MODELS
from . import YOLOv8CSPDarknet
from ..layers import SPPFBottleneck, build_lite_series
from ..utils import make_divisible
import torch.nn as nn


@MODELS.register_module()
class LiteBackbone(YOLOv8CSPDarknet):

    def __init__(self,
                 lite_arch_setting: List[dict],
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.lite_arch_setting = lite_arch_setting
        super().__init__(**kwargs, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return DepthwiseSeparableConvModule(
                self.input_channels,
                make_divisible(self.arch_setting[0][0], self.widen_factor),
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                dw_act_cfg=None,
                act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, _, _, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        stage = []
        conv_layer = DepthwiseSeparableConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        lite_block_cfg = self.lite_arch_setting[stage_idx]
        lite_block_cfg.update(norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        stage.append(build_lite_series(block_cfg=lite_block_cfg,
                                          in_channels=out_channels,
                                          out_channels=out_channels))
        if use_spp:
            spp = LiteSPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage


class LiteSPPFBottleneck(SPPFBottleneck):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(in_channels, out_channels, kernel_sizes,
                 use_conv_first, mid_channels_scale, conv_cfg,
                 norm_cfg, act_cfg, init_cfg)

        mid_channels = int(in_channels * mid_channels_scale) \
            if use_conv_first else in_channels
        conv2_in_channels = mid_channels * 4 \
            if isinstance(kernel_sizes, int) \
            else mid_channels * (len(kernel_sizes) + 1)

        if self.conv1 is not None:
            self.conv1 = DepthwiseSeparableConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
