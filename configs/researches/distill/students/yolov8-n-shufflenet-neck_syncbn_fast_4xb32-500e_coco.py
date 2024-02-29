_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw.py',
          './yolov8-n-shufflenet-neck.py']

val_cfg = dict(type='ValLoop')

num_classes = 80
data_root = 'data/coco/'
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'  # Prefix of train image path
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'  # Prefix of val image path
train_batch_size_per_gpu = 32
max_epochs = 500

model = _base_.architecture
model.bbox_head.head_module.num_classes = num_classes
model.train_cfg.assigner.num_classes = num_classes
del _base_.architecture

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(data_root=data_root,
                 ann_file=train_ann_file,
                 data_prefix=dict(img=train_data_prefix)))
del _base_.train_dataloader.dataset.metainfo
del _base_.val_dataloader.dataset.metainfo
del _base_.test_dataloader.dataset.metainfo
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file))
test_dataloader = val_dataloader

_base_.val_evaluator.ann_file = data_root + val_ann_file
del _base_.val_evaluator.classwise
_base_.test_evaluator.ann_file = data_root + val_ann_file
del _base_.test_evaluator.classwise

save_epoch_intervals=10
val_interval=save_epoch_intervals
default_hooks = dict(
    param_scheduler=dict(
        max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals))
_base_.custom_hooks[1].switch_epoch = max_epochs - _base_.close_mosaic_epochs

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[((max_epochs - _base_.close_mosaic_epochs),
                        _base_.val_interval_stage2)])

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

del _base_.class_name
del _base_.metainfo

