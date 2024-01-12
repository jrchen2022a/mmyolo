_base_ = './yolov8_n_syncbn_fast_8xb16-500e_coco.py'
train_batch_size_per_gpu = 2
train_num_workers = 1

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = _base_.work_dir_root+'/work_dirs/temp/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend')    ])
