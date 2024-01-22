_base_ = './yolov8_l_mask-refine_syncbn_fast_4xb16-200e_dw.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=3,
        selector_type='SelectorCSPLayerWithTwoConv'))
