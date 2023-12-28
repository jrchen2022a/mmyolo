_base_ = './yolov8_n_syncbn_fast_4xb16-500e_coco.py'

work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/coco/yolov8_n_syncbn_fast_4xb16-500e_coco_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_n_fuser'))
    ])
# work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#         dict(type='WandbVisBackend', init_kwargs=dict(project='temp', name='yolov8_n_selector'))
#     ])
# _base_.wandb_project_name
# 'temp'
model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))