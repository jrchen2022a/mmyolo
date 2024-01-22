_base_ = '../../yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw.py'
# _base_ = '../algorithm_agnostic/debug.py'
wandb_project_name = 'distill_exp'
model = dict(
    backbone=dict(
        type='DepthSeparableBackbone',
        version='v2'))

work_dir = (_base_.work_dir_root+'/work_dirs/{0}/yolov8-n-dscv2_4xb16_200e_dw/'
            .format(wandb_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=wandb_project_name, name='yolov8-n-dscv2'))
    ])