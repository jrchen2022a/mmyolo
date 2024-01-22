_base_ = './dw_runtime_v8_500e.py'

base_lr = 0.01
weight_decay = 0.0005

val_interval = _base_.save_epoch_intervals
val_interval_stage2 = 1

param_scheduler = None

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=_base_.train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=_base_.data_root + _base_.val_ann_file,
    classwise=True,
    metric='bbox')
test_evaluator = val_evaluator


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=_base_.max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[((_base_.max_epochs - _base_.close_mosaic_epochs),
                        val_interval_stage2)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
