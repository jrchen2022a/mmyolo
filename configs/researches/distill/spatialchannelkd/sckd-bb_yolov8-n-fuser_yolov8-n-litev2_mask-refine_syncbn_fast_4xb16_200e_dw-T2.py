_base_ = 'sckd-bb_yolov8-n-fuser_yolov8-n-litev2_mask-refine_syncbn_fast_4xb16_200e_dw.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

tau = 2
_base_.model.distiller.distill_losses.loss_s2.tau = tau
_base_.model.distiller.distill_losses.loss_s3.tau = tau
_base_.model.distiller.distill_losses.loss_s4.tau = tau
