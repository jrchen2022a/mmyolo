# 占用20G / 3090 20G 1:39day 4:13day / 4090 20G 1:23day 4:7day
#       / A5000 20G 1:40day 4:12day / A40 20G 1:36day / V100 20G 1:49day 4:17day
_base_ = './yolov8_l_syncbn_fast_8xb16-500e_coco.py'
train_batch_size_per_gpu = 8
train_num_workers = 4

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov8_l_syncbn_fast_4xb8-500e_coco_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_l_fuser'))
    ])
# work_dir = _base_.work_dir_root+'/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))
