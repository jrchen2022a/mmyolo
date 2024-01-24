_base_ = './pkd-backbone_yolov8-n-fuser_yolov8-n-dscv2_4xb16_200e_dw-uc.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

_base_.model.architecture.init_cfg = dict(
    type='Pretrained', checkpoint='checkpoints/yolov8-n-dscv2_best-dw.pth')

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='PKDLoss', loss_weight=256),
            loss_s3=dict(type='PKDLoss', loss_weight=256),
            loss_s4=dict(type='PKDLoss', loss_weight=256))))
