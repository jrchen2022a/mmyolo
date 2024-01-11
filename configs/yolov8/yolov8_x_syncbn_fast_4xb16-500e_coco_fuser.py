# A100 40G  1:43day / A40 1:49day
_base_ = './yolov8_x_syncbn_fast_8xb16-500e_coco.py'

work_dir = _base_.work_dir_root+'/work_dirs/coco/yolov8_x_syncbn_fast_4xb16-500e_coco_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_x_fuser'))
    ])
# work_dir = _base_.work_dir_root+'/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')    ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))
