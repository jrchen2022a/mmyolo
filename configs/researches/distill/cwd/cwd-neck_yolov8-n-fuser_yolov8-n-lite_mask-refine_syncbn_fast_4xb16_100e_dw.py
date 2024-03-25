_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_100e_dw_mask-refine.py',
          '../algorithm_agnostic/distill-neck_yolov8-n-fuser_yolov8-n-lite.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='ChannelWiseDivergence', tau=4, loss_weight=10),
            loss_s3=dict(type='ChannelWiseDivergence', tau=4, loss_weight=10),
            loss_s4=dict(type='ChannelWiseDivergence', tau=4, loss_weight=10))))

#_base_.custom_hooks.append(dict(type='DistillWeightHook', _scope_='mmrazor'))
