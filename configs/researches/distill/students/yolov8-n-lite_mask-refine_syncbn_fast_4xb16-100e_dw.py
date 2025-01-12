_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_100e_dw_mask-refine.py',
          './yolov8-n-lite.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

val_cfg = dict(type='ValLoop')
model = _base_.architecture
del _base_.architecture
