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

work_dir = (_base_.work_dir_root+'/work_dirs/{0}/fbkd_yolov8-n_yolov8-n-dscv2_snoloss_4xb16_200e_dw.py/'
            .format(_base_.wandb_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=_base_.wandb_project_name, name='yolov8_n_fbkd_dscv2_snoloss'))
    ])
# work_dir = _base_.work_dir_root+'/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')
#     ])
# train_batch_size_per_gpu = 2
# train_dataloader = dict(
#     batch_size=train_batch_size_per_gpu)
# optim_wrapper = dict(
#     optimizer=dict(
#         batch_size_per_gpu=_base_.train_batch_size_per_gpu))

stages_output_channels = {
    'n': [32,  64,  128, 256],
    's': [64,  128, 256, 512],
    'm': [96,  192, 384, 576],
    'l': [128, 256, 512, 512],
    'x': [160, 320, 640, 640]
}
hc_channels = _base_.num_classes
hr_channels = 4 * 16

teacher_ckpt = 'work_dirs/dianwang/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser/20231218_024819*/best_coco/off_precision_epoch_175.pth'
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::researches/distill/students/yolov8-n-dscv2_syncbn_fast_4xb16-200e_dw.py',
        pretrained=False),
    calculate_student_loss=False,
    teacher=dict(
        cfg_path='mmyolo::researches/yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            stage_s2=dict(type='ModuleInputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleInputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleInputs', source='neck.reduce_layers.2'),
            stage_hcs2=dict(type='ModuleOutputs', source='bbox_head.head_module.cls_preds.0.2'),
            stage_hcs3=dict(type='ModuleOutputs', source='bbox_head.head_module.cls_preds.1.2'),
            stage_hcs4=dict(type='ModuleOutputs', source='bbox_head.head_module.cls_preds.2.2'),
            stage_hrs2=dict(type='ModuleOutputs', source='bbox_head.head_module.reg_preds.0.2'),
            stage_hrs3=dict(type='ModuleOutputs', source='bbox_head.head_module.reg_preds.1.2'),
            stage_hrs4=dict(type='ModuleOutputs', source='bbox_head.head_module.reg_preds.2.2')),
        teacher_recorders=dict(
            stage_s2=dict(type='ModuleInputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleInputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleInputs', source='neck.reduce_layers.2'),
            stage_hcs2=dict(type='ModuleOutputs', source='bbox_head.head_module.cls_preds.0.2'),
            stage_hcs3=dict(type='ModuleOutputs', source='bbox_head.head_module.cls_preds.1.2'),
            stage_hcs4=dict(type='ModuleOutputs', source='bbox_head.head_module.cls_preds.2.2'),
            stage_hrs2=dict(type='ModuleOutputs', source='bbox_head.head_module.reg_preds.0.2'),
            stage_hrs3=dict(type='ModuleOutputs', source='bbox_head.head_module.reg_preds.1.2'),
            stage_hrs4=dict(type='ModuleOutputs', source='bbox_head.head_module.reg_preds.2.2')),
        distill_losses=dict(
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss'),
            loss_s4=dict(type='FBKDLoss'),
            loss_hcs2=dict(type='FBKDLoss'),
            loss_hcs3=dict(type='FBKDLoss'),
            loss_hcs4=dict(type='FBKDLoss'),
            loss_hrs2=dict(type='FBKDLoss'),
            loss_hrs3=dict(type='FBKDLoss'),
            loss_hrs4=dict(type='FBKDLoss')),
        connectors=dict(
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
                sub_sample=True),
            loss_hcs2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=hc_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hcs2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=hc_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hcs3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=hc_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hcs3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=hc_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hcs4_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=hc_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hcs4_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=hc_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hrs2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=hr_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hrs2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=hr_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hrs3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=hr_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hrs3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=hr_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hrs4_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=hr_channels,
                mode='dot_product',
                sub_sample=True),
            loss_hrs4_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=hr_channels,
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
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
                    data_idx=0)),
            loss_hcs2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_hcs2',
                    connector='loss_hcs2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_hcs2',
                    connector='loss_hcs2_tfeat')),
            loss_hcs3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_hcs3',
                    connector='loss_hcs3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_hcs3',
                    connector='loss_hcs3_tfeat')),
            loss_hcs4=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_hcs4',
                    connector='loss_hcs4_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_hcs4',
                    connector='loss_hcs4_tfeat')),
            loss_hrs2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_hrs2',
                    connector='loss_hrs2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_hrs2',
                    connector='loss_hrs2_tfeat')),
            loss_hrs3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_hrs3',
                    connector='loss_hrs3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_hrs3',
                    connector='loss_hrs3_tfeat')),
            loss_hrs4=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_hrs4',
                    connector='loss_hrs4_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_hrs4',
                    connector='loss_hrs4_tfeat')))))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
