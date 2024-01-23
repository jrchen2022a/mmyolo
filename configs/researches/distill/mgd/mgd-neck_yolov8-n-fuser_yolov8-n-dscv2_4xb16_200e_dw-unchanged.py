# _base_ = '../algorithm_agnostic/debug.py'
_base_ = './mgd-neck_yolov8-n-fuser_yolov8-n-dscv2_4xb16_200e_dw.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'


del _base_.train_cfg['dynamic_intervals']
del _base_.custom_hooks[1]
del _base_.val_interval_stage2
del _base_.train_pipeline
del _base_.train_pipeline_stage2
