# _base_ = '../algorithm_agnostic/debug.py'
_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw.py',
          '../algorithm_agnostic/distill-bb-neck_yolov8-n-fuser_yolov8-n-dscv2.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='PKDLoss', loss_weight=50),
            loss_s3=dict(type='PKDLoss', loss_weight=50),
            loss_s4=dict(type='PKDLoss', loss_weight=50),
            loss_ns2=dict(type='PKDLoss', loss_weight=100),
            loss_ns3=dict(type='PKDLoss', loss_weight=100),
            loss_ns4=dict(type='PKDLoss', loss_weight=100))))

del _base_.train_cfg['dynamic_intervals']
del _base_.custom_hooks[1]
del _base_.val_interval_stage2
del _base_.train_pipeline
del _base_.train_pipeline_stage2
