_base_ = './yolov8-n.py'

architecture = dict(
    backbone=dict(
        type='DepthSeparableBackbone',
        version='v2'))
