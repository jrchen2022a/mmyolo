_base_ = './yolov8-n.py'

_SHUFFLE_BLOCK = dict(type='ShuffleBlock')
bb_arch_setting = [
    # stage 1-4 的模块 [num, block_type]
    [1, _SHUFFLE_BLOCK],
    [2, _SHUFFLE_BLOCK],
    [2, _SHUFFLE_BLOCK],
    [1, _SHUFFLE_BLOCK]
]
neck_arch_setting = [1, _SHUFFLE_BLOCK]

architecture = dict(
    backbone=dict(
        type='LiteBackbone',
        lite_arch_setting=bb_arch_setting),
    neck=dict(
        type='LitePAFPN',
        lite_arch_setting=neck_arch_setting),
    bbox_head=dict(
        head_module=dict(
            type='LiteYOLOv8HeadModule')))

del bb_arch_setting
del neck_arch_setting
del _SHUFFLE_BLOCK
