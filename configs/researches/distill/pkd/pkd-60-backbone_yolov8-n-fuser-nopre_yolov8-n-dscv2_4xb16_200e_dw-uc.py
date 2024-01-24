_base_ = './pkd-60-backbone_yolov8-n-fuser_yolov8-n-dscv2_4xb16_200e_dw-uc.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

_base_.model.type = 'SingleTeacherDistill'
_base_.model.teacher_ckpt = None
_base_.model.teacher_trainable = True
_base_.model.teacher_norm_eval = False
