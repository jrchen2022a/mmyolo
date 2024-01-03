#
_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

base_lr = 0.005
optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr))
train_num_workers = 4
train_dataloader = dict(
    num_workers=train_num_workers)

work_dir = './work_dirs/coco/yolov8_s_syncbn_fast_4xb16-500e_coco_fuser2/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_s_fuser2'))
    ])
# work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#     ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=2,
        selector_type='SelectorCSPLayerWithTwoConv'))