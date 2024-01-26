from typing import List, Tuple, Union

from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from . import YOLOv8CSPDarknet
from ..layers import SPPFBottleneck
from ..utils import make_divisible


@MODELS.register_module()
class ShuffleNetBackbone(YOLOv8CSPDarknet):

    def __init__(self,
                 shuffle_arch_setting: List[dict],
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.shuffle_arch_setting = shuffle_arch_setting
        super().__init__(
            arch=arch,
            last_stage_out_channels=last_stage_out_channels,
            plugins=plugins,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)


    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, _, _, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        stage = []
        num_blocks, shuffle_block_cfg = self.shuffle_arch_setting[stage_idx]
        for i in range(num_blocks):
            if i == 0 : # 入口处降采样 & 通道收缩
                shuffle_block_cfg.update(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2)
            else:
                shuffle_block_cfg.update(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1)
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
