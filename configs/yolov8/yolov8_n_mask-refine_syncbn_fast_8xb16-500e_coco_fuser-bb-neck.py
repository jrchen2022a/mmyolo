_base_ = './yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py'

work_dir = _base_.work_dir_root+'/work_dirs/coco/{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends.append(
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='research_coco',
                          name='yolov8_n_fuser-bb-neck_mr')))

model = dict(
    backbone=dict(
            type='YOLOv8SelectorCSPDarknet',
            num_selectors=3,
            selector_type='SelectorCSPLayerWithTwoConv'),
    neck=dict(
        type='YOLOv8SelectorPAFPN',
        num_selectors=3))
