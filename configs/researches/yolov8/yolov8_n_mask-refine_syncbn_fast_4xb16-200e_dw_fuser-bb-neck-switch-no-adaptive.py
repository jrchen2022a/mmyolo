"""
    聚合器自适应机制取消 消融实验
"""
_base_ = './yolov8_n_mask-refine_syncbn_fast_4xb16-200e_dw.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name='{{fileBasenameNoExtension}}'
# work_dir = _base_.work_dir_root+'/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')
#     ])

model = dict(
    backbone=dict(
            type='YOLOv8SelectorCSPDarknet',
            num_selectors=3,
            switch_adaptive=False,
            selector_type='SelectorCSPLayerWithTwoConv'),
    neck=dict(
        type='YOLOv8SelectorPAFPN',
        switch_adaptive=False,
        num_selectors=3))
