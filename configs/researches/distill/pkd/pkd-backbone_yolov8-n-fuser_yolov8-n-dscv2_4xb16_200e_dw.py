_base_ = '../algorithm_agnostic/distill-backbone_yolov8-n-fuser_yolov8-n-dscv2_8xb16_200e_dw.py'

work_dir = (_base_.work_dir_root + '/work_dirs/{0}/{{fileBasenameNoExtension}}/'
            .format(_base_.wandb_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project=_base_.wandb_project_name, name='{{fileBasenameNoExtension}}'))
    ])

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='PKDLoss', loss_weight=6),
            loss_s3=dict(type='PKDLoss', loss_weight=6),
            loss_s4=dict(type='PKDLoss', loss_weight=6))))
