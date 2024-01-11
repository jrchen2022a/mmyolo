_base_ = './yolov8_s_syncbn_fast_4xb16-200e_dw.py'

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

work_dir = _base_.work_dir_root+'/work_dirs/dianwang/yolov8_n_syncbn_fast_'+str(_base_.nGPU)+'xb'+str(_base_.train_batch_size_per_gpu)+'-'+str(_base_.max_epochs)+'e_dw/'
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',init_kwargs=dict(_base_.wandb_init_kwargs,name='yolov8_n'))
    ],
    name='visualizer')

