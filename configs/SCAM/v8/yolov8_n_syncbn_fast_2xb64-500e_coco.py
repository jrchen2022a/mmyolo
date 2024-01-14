_base_ = './yolov8_n_syncbn_fast_8xb16-500e_coco.py'

train_batch_size_per_gpu = 64
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov8_n_syncbn_fast_2xb64-500e_coco_SCAM/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='coco', name='yolov8_n_scam'))
    ])