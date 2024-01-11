# 22G 3090 1:28day
_base_ = './yolov8_m_syncbn_fast_8xb16-500e_coco.py'
base_lr = 0.0025
optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr))
work_dir = './work_dirs/coco/yolov8_m_syncbn_fast_1xb16-500e_coco_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_m_fuser'))
    ])
# work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')     ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))