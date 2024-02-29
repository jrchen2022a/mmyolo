_base_ = '../../_base_/dw_schedule_v8_8xb16_500e.py'

wandb_project_name = 'distill_exp3'
_base_.visualizer.vis_backends[1].init_kwargs.project = wandb_project_name
work_dir_root = _base_.save_work_dir_root + wandb_project_name + '/'

max_epochs = 200
_base_.default_hooks.param_scheduler.max_epochs = max_epochs
_base_.custom_hooks[1].switch_epoch = max_epochs - _base_.close_mosaic_epochs
_base_.train_cfg.max_epochs = max_epochs
_base_.train_cfg.dynamic_intervals = [((max_epochs - _base_.close_mosaic_epochs), _base_.val_interval_stage2)]

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
