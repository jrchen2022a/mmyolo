_base_ = './pkd-neck_yolov8-n-fuser_yolov8-n-shufflenet-neck_4xb16_200e_dw-uc-5-25-50.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

_base_.custom_hooks.append(dict(type='StopDistillHook', _scope_='mmrazor', stop_epoch=50))
