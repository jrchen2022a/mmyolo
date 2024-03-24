from typing import List

from mmdet.utils import ConfigType
from mmcv.cnn import DepthwiseSeparableConvModule
from mmyolo.registry import MODELS
from . import YOLOv8CSPDarknet
from ..layers import SPPFBottleneck, build_shuffle_series
from ..utils import make_divisible
import torch.nn as nn


@MODELS.register_module()
class LiteBackbone(YOLOv8CSPDarknet):

    def __init__(self,
                 shuffle_arch_setting: List[dict],
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.shuffle_arch_setting = shuffle_arch_setting
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
        num_blocks, shuffle_block_cfg = self.shuffle_arch_setting[stage_idx]
        shuffle_block_cfg.update(norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        stage.append(build_shuffle_series(block_cfg=shuffle_block_cfg,
                                          num_blocks=num_blocks,
                                          in_channels=out_channels,
                                          out_channels=out_channels))
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage
