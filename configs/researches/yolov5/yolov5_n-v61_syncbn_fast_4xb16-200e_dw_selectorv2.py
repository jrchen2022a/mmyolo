_base_ = './yolov5_n-v61_syncbn_fast_4xb16-200e_dw.py'

work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/dianwang/yolov5_n-v61_syncbn_fast_' + str(_base_.nGPU) + 'xb' + str(
    _base_.train_batch_size_per_gpu) + '-' + str(_base_.max_epochs) + 'e_dw_selectorv2/'
# work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/temp/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(_base_.wandb_init_kwargs, name='yolov5_n_selectorv2'))
    ])
#,project='temp'
# _base_.wandb_init_kwargs
model = dict(
    backbone=dict(
        type='YOLOv5SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerV2'))
