_base_ = './yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py'

train_batch_size_per_gpu = 128
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov5_n-v61_syncbn_fast_2xb128-300e_coco_SCAM_2/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='coco', name='yolov5_n_scam_2'))
    ])