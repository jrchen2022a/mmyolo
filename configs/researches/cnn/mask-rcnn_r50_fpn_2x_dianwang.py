# 这个不用跑，没有mask
_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'datasets_dianwang.py',
    'schedule_2x_dianwang.py',
    'runtime_dianwang.py'
]

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name='{{fileBasenameNoExtension}}'

# del _base_.model.backbone.init_cfg
_base_.model.roi_head.bbox_head.num_classes = _base_.num_classes
_base_.model.roi_head.mask_head.num_classes = _base_.num_classes
