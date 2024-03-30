_base_ = './yolov8-n.py'

conv_groups = 4
chunk_even = True

bb_arch_setting = [
    dict(type='LiteBlock', num_block=1, conv_groups=conv_groups, chunk_even=chunk_even),
    dict(type='LiteBlock', num_block=2, conv_groups=conv_groups, chunk_even=chunk_even),
    dict(type='LiteBlock', num_block=2, conv_groups=conv_groups, chunk_even=chunk_even),
    dict(type='LiteBlock', num_block=1, conv_groups=conv_groups, chunk_even=chunk_even)]
neck_arch_setting = dict(type='LiteBlock', num_block=1, conv_groups=conv_groups, chunk_even=chunk_even)

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

