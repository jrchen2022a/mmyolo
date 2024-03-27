_base_ = './yolov8-n.py'

bb_arch_setting = [
    dict(type='LiteBlock', num_block=1, conv_groups=4),
    dict(type='LiteBlock', num_block=2, conv_groups=4),
    dict(type='LiteBlock', num_block=2, conv_groups=4),
    dict(type='LiteBlock', num_block=1, conv_groups=4)]
neck_arch_setting = dict(type='LiteBlock', num_block=1, conv_groups=4)

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

