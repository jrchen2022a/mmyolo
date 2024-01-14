_base_ = './yolov6_n_syncbn_fast_8xb32-400e_coco.py'

train_batch_size_per_gpu = 128
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov6_n_syncbn_fast_2x128-400e_coco_SCAM/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='coco', name='yolov6_n_scam'))
    ])