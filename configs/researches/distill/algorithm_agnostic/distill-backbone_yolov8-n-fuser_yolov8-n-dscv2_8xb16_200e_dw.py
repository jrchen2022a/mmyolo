_base_ = ['../../_base_/schedule_v8_8xb16_500e.py', '../students/yolov8-n-dscv2.py']
wandb_project_name = 'distill_exp'
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

stages_output_channels = {
    'n': [32, 64, 128, 256],
    's': [64, 128, 256, 512],
    'm': [96, 192, 384, 576],
    'l': [128, 256, 512, 512],
    'x': [160, 320, 640, 640]
}

teacher_ckpt = 'work_dirs/dianwang/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser/20231218_024819*/best_coco/off_precision_epoch_175.pth'
student = _base_.model
student.neck.init_cfg = dict(
    type='Pretrained', prefix='neck.', checkpoint=teacher_ckpt)
student.bbox_head.init_cfg = dict(
    type='Pretrained', prefix='bbox_head.', checkpoint=teacher_ckpt)
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=student,
    teacher=dict(
        cfg_path='mmyolo::researches/yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            stage_s2=dict(type='ModuleInputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleInputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleInputs', source='neck.reduce_layers.2')),
        teacher_recorders=dict(
            stage_s2=dict(type='ModuleInputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleInputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleInputs', source='neck.reduce_layers.2')),
        loss_forward_mappings=dict(
            loss_s2=dict(
                preds_S=dict(from_student=True,  recorder='stage_s2', data_idx=0),
                preds_T=dict(from_student=False, recorder='stage_s2', data_idx=0)),
            loss_s3=dict(
                preds_S=dict(from_student=True,  recorder='stage_s3', data_idx=0),
                preds_T=dict(from_student=False, recorder='stage_s3', data_idx=0)),
            loss_s4=dict(
                preds_S=dict(from_student=True,  recorder='stage_s4', data_idx=0),
                preds_T=dict(from_student=False, recorder='stage_s4', data_idx=0)))))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
