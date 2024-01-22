# 当你想调试时，_base_引入这个文件，并更改如下_base_
_base_ = '../../yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw.py'

train_batch_size_per_gpu = 2
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = 'work_dirs/temp/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend')
    ])