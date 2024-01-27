from typing import List

from mmdet.utils import ConfigType

from mmyolo.registry import MODELS
from . import YOLOv8CSPDarknet
from ..layers import SPPFBottleneck
from ..utils import make_divisible


@MODELS.register_module()
class ShuffleNetBackbone(YOLOv8CSPDarknet):

    def __init__(self,
                 shuffle_arch_setting: List[dict],
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.shuffle_arch_setting = shuffle_arch_setting
        super().__init__(**kwargs, norm_cfg=norm_cfg, act_cfg=act_cfg)


    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, _, _, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        stage = []
        num_blocks, shuffle_block_cfg = self.shuffle_arch_setting[stage_idx]
        shuffle_block_cfg.update(out_channels=out_channels,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
        for i in range(num_blocks):
            if i == 0 : # 入口处降采样 & 通道收缩
                shuffle_block_cfg.update(in_channels=in_channels, stride=2)
            else:
                shuffle_block_cfg.update(in_channels=out_channels, stride=1)
            stage.append(MODELS.build(shuffle_block_cfg))

        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage
