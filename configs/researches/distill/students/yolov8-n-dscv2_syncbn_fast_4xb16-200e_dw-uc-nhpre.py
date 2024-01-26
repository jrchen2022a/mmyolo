_base_ = './yolov8-n-dscv2_syncbn_fast_4xb16-200e_dw.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

del _base_.train_cfg['dynamic_intervals']
del _base_.custom_hooks[1]
del _base_.val_interval_stage2
del _base_.train_pipeline
del _base_.train_pipeline_stage2

model = _base_.architecture
teacher_ckpt = 'checkpoints/yolov8-n-fuser_best-dw.pth'
model.neck.init_cfg = dict(
    type='Pretrained', prefix='neck.', checkpoint=teacher_ckpt)
model.bbox_head.init_cfg = dict(
    type='Pretrained', prefix='bbox_head.', checkpoint=teacher_ckpt)
