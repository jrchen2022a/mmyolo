_base_ = [
    'mmdet::_base_/models/retinanet_r50_fpn.py',
    'datasets_dianwang.py',
    'schedule_2x_dianwang.py',
    'runtime_dianwang.py'
]

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name='{{fileBasenameNoExtension}}'


_base_.model.bbox_head.num_classes = _base_.num_classes

optim_wrapper = dict(
    optimizer=dict(lr=0.002, momentum=0.9, weight_decay=0.0001))
