_base_ = './pkd-backbone_yolov8-n-fuser_yolov8-n-dscv2_4xb16_200e_dw-uc.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

_base_.model.architecture.neck.init_cfg = dict(
    type='Pretrained', prefix='neck.', checkpoint=_base_.teacher_ckpt)
_base_.model.architecture.bbox_head.init_cfg = dict(
    type='Pretrained', prefix='bbox_head.', checkpoint=_base_.teacher_ckpt)
