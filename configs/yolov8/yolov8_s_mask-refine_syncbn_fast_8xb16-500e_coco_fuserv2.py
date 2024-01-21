#
_base_ = './yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

# work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_fuserv2/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#         dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_s_fuserv2_mr'))
#     ])

train_batch_size_per_gpu = 2
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))
work_dir = _base_.work_dir_root+'/work_dirs/temp/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        version=2,
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))