_base_ = 'yolov8-n-litev2_mask-refine_syncbn_fast_4xb16-200e_dw.py'

dw_mode = True
chunk_even = False

_base_.model.backbone.lite_arch_setting[0].chunk_even = chunk_even
_base_.model.backbone.lite_arch_setting[1].chunk_even = chunk_even
_base_.model.backbone.lite_arch_setting[2].chunk_even = chunk_even
_base_.model.backbone.lite_arch_setting[3].chunk_even = chunk_even
_base_.model.neck.lite_arch_setting.chunk_even = chunk_even

_base_.model.backbone.lite_arch_setting[0].dw_mode = dw_mode
_base_.model.backbone.lite_arch_setting[1].dw_mode = dw_mode
_base_.model.backbone.lite_arch_setting[2].dw_mode = dw_mode
_base_.model.backbone.lite_arch_setting[3].dw_mode = dw_mode
_base_.model.neck.lite_arch_setting.dw_mode = dw_mode

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

