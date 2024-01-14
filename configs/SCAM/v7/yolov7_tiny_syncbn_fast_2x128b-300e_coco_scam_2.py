_base_ = './yolov7_tiny_syncbn_fast_8x16b-300e_coco.py'

train_batch_size_per_gpu = 128
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

model = dict(
    backbone=dict(
        scam_insert_idx=2))

work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov7_tiny_syncbn_fast_2x128b-300e_coco_SCAM_2/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='coco', name='yolov7_tiny_scam_2'))
    ])