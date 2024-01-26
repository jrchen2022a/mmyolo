_base_ = './yolov8-n.py'

_SHUFFLE_BLOCK = dict(type='ShuffleBlock', kernel_size=3)
arch_setting = [
    # stage 1-4 的模块 [num, block_type]
    [1, _SHUFFLE_BLOCK],
    [1, _SHUFFLE_BLOCK],
    [2, _SHUFFLE_BLOCK],
    [1, _SHUFFLE_BLOCK]
]

architecture = dict(
    backbone=dict(
        type='ShuffleNetBackbone',
        shuffle_arch_setting=arch_setting))

del arch_setting
del _SHUFFLE_BLOCK
