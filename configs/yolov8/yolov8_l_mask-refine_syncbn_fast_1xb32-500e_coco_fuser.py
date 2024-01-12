#
_base_ = './yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py'
base_lr = 0.01
optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr))
train_batch_size_per_gpu = 32

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))
work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov8_l_mask-refine_syncbn_fast_4xb16-500e_coco_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_l_fuser_mr'))
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
