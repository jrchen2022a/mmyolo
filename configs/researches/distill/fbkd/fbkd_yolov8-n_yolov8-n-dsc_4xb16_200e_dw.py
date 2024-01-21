_base_ = '../../_base_/schedule_v8_8xb16_500e.py'

max_epochs = 200
default_hooks = dict(
    param_scheduler=dict(
        max_epochs=max_epochs))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    dynamic_intervals=[((max_epochs - _base_.close_mosaic_epochs),
                        _base_.val_interval_stage2)])

work_dir = (_base_.work_dir_root+'/work_dirs/{0}/fbkd_yolov8-n_yolov8-n-dsc_4xb16_200e_dw.py/'
            .format(_base_.wandb_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=_base_.wandb_project_name, name='yolov8_n_fbkd_dsc'))
    ])
# work_dir = _base_.work_dir_root+'/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')
#     ])

stages_output_channels = {
    'n': [32,  64,  128, 256],
    's': [64,  128, 256, 512],
    'm': [96,  192, 384, 576],
    'l': [128, 256, 512, 512],
    'x': [160, 320, 640, 640]
}

teacher_ckpt = 'work_dirs/dianwang/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser/20231218_024819*/epoch_200.pth'
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::researches/distill/students/yolov8-n-dsc_syncbn_fast_4xb16-200e_dw.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmyolo::researches/yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            stage_s1=dict(type='ModuleInputs', source='backbone.stage2.0.conv'),
            stage_s2=dict(type='ModuleInputs', source='backbone.stage3.0.conv'),
            stage_s3=dict(type='ModuleInputs', source='backbone.stage4.0.conv'),
            stage_s4=dict(type='ModuleInputs', source='backbone.stage4.2.conv1.conv')),
        teacher_recorders=dict(
            stage_s1=dict(type='ModuleInputs', source='backbone.stage2.0.conv'),
            stage_s2=dict(type='ModuleInputs', source='backbone.stage3.0.conv'),
            stage_s3=dict(type='ModuleInputs', source='backbone.stage4.0.conv'),
            stage_s4=dict(type='ModuleInputs', source='backbone.stage4.2.conv1.conv')),
        distill_losses=dict(
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss'),
            loss_s4=dict(type='FBKDLoss')),
        connectors=dict(
            loss_s1_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=stages_output_channels['n'][0],
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s1_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=stages_output_channels['n'][0],
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=stages_output_channels['n'][1],
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=stages_output_channels['n'][1],
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=stages_output_channels['n'][2],
                mode='dot_product',
                sub_sample=True),
            loss_s3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=stages_output_channels['n'][2],
                mode='dot_product',
                sub_sample=True),
            loss_s4_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=stages_output_channels['n'][3],
                mode='dot_product',
                sub_sample=True),
            loss_s4_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=stages_output_channels['n'][3],
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
            loss_s1=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s1',
                    connector='loss_s1_sfeat',
                    data_idx=0),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s1',
                    connector='loss_s1_tfeat',
                    data_idx=0)),
            loss_s2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s2',
                    connector='loss_s2_sfeat',
                    data_idx=0),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s2',
                    connector='loss_s2_tfeat',
                    data_idx=0)),
            loss_s3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s3',
                    connector='loss_s3_sfeat',
                    data_idx=0),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s3',
                    connector='loss_s3_tfeat',
                    data_idx=0)),
            loss_s4=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s4',
                    connector='loss_s4_sfeat',
                    data_idx=0),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s4',
                    connector='loss_s4_tfeat',
                    data_idx=0)))))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
