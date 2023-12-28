# 跑不动，真的跑不动，72天
_base_ = './yolov8_x_syncbn_fast_4xb8-500e_coco.py'

work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/coco/yolov8_x_syncbn_fast_4xb8-500e_coco_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_x_fuser'))
    ])
# work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')    ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))
