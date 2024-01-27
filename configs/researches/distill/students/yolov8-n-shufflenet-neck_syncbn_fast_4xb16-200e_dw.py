_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw.py',
          './yolov8-n-shufflenet-neck.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

val_cfg = dict(type='ValLoop')
model = _base_.architecture