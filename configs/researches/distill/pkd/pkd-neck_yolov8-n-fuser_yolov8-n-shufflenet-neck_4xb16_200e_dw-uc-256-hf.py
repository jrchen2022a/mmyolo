_base_ = './pkd-neck_yolov8-n-fuser_yolov8-n-shufflenet-neck_4xb16_200e_dw-uc-50.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
	architecture=dict(bbox_head=dict(type='FreezableYOLOv8Head', frozen_head=True)),
	distiller=dict(
	        distill_losses=dict(
	            loss_s2=dict(type='PKDLoss', loss_weight=256),
	            loss_s3=dict(type='PKDLoss', loss_weight=256),
	            loss_s4=dict(type='PKDLoss', loss_weight=256))))

