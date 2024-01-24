#
_base_ = './yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py'

work_dir = _base_.work_dir_root+'/work_dirs/coco/{{fileBasenameNoExtension}}/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_x_fuserv2_mr'))
    ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        version=2,
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))