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

work_dir = (_base_.work_dir_root + '/work_dirs/{0}/cwd-backbone_yolov8-n_yolov8-n-dscv2_4xb16_200e_dw/'
            .format(_base_.wandb_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project=_base_.wandb_project_name, name='cwd-backbone_yolov8-n_yolov8-n-dscv2'))
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
    'n': [32, 64, 128, 256],
    's': [64, 128, 256, 512],
    'm': [96, 192, 384, 576],
    'l': [128, 256, 512, 512],
    'x': [160, 320, 640, 640]
}

teacher_ckpt = 'work_dirs/dianwang/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser/20231218_024819*/best_coco/off_precision_epoch_175.pth'
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::researches/distill/students/yolov8-n-dscv2_syncbn_fast_4xb16-200e_dw.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmyolo::researches/yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            stage_s2=dict(type='ModuleOutputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleOutputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleOutputs', source='neck.reduce_layers.2')),
        teacher_recorders=dict(
            stage_s2=dict(type='ModuleOutputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleOutputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleOutputs', source='neck.reduce_layers.2')),
        distill_losses=dict(
            loss_s2=dict(type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_s3=dict(type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_s4=dict(type='ChannelWiseDivergence', tau=1, loss_weight=10)),
        loss_forward_mappings=dict(
            loss_s2=dict(
                preds_S=dict(from_student=True,  recorder='stage_s2'),
                preds_T=dict(from_student=False, recorder='stage_s2')),
            loss_s3=dict(
                preds_S=dict(from_student=True,  recorder='stage_s3'),
                preds_T=dict(from_student=False, recorder='stage_s3')),
            loss_s4=dict(
                preds_S=dict(from_student=True,  recorder='stage_s4'),
                preds_T=dict(from_student=False, recorder='stage_s4')))))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
