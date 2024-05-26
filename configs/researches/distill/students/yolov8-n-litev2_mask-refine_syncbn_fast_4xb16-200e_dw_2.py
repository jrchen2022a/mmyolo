_base_ = 'yolov8-n-litev2_mask-refine_syncbn_fast_4xb16-200e_dw.py'

conv_groups = 2

_base_.model.backbone.lite_arch_setting[0].conv_groups = conv_groups
_base_.model.backbone.lite_arch_setting[1].conv_groups = conv_groups
_base_.model.backbone.lite_arch_setting[2].conv_groups = conv_groups
_base_.model.backbone.lite_arch_setting[3].conv_groups = conv_groups
_base_.model.neck.lite_arch_setting.conv_groups = conv_groups

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'
