_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw_mask-refine.py',
          '../algorithm_agnostic/distill-neck_yolov8-n-fuser_yolov8-n-lite.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='SpatialChannelWiseDivergence', tau=1, loss_weight=20),
            loss_s3=dict(type='SpatialChannelWiseDivergence', tau=1, loss_weight=20),
            loss_s4=dict(type='SpatialChannelWiseDivergence', tau=1, loss_weight=20))))

del _base_.custom_hooks[-1]
_base_.custom_hooks.append(dict(type='DistillWeightHook', _scope_='mmrazor', start_epoch=80, stop_epoch=160))
_base_.custom_hooks.append(dict(type='StopDistillHook', _scope_='mmrazor', stop_epoch=160))