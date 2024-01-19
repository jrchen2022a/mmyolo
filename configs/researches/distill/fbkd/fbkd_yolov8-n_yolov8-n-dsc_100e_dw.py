_base_ = [
    '../../_base_/datasets/datasets_dianwang.py',
    '../../_base_/default_runtime.py'
]

base_lr = 0.01
weight_decay = 0.0005
train_batch_size_per_gpu = 16
train_num_workers = 8
persistent_workers = True
lr_factor = 0.01
max_epochs = 100
max_keep_ckpts = 3
env_cfg = dict(cudnn_benchmark=True)

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=_base_.save_epoch_intervals)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=_base_.save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='yolov5_collate'),
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=_base_.metainfo,
        data_root=_base_.data_root,
        ann_file=_base_.train_ann_file,
        data_prefix=dict(img=_base_.train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=_base_.data_root,
        metainfo=_base_.metainfo,
        test_mode=True,
        data_prefix=dict(img=_base_.val_data_prefix),
        ann_file=_base_.val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=_base_.data_root + _base_.val_ann_file,
    classwise=True,
    metric='bbox')
test_evaluator = val_evaluator


teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
        pretrained=True),
    teacher=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv')),
        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv')),
        distill_losses=dict(
            loss_s0=dict(type='FBKDLoss'),
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss')),
        connectors=dict(
            loss_s0_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s0_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s1_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s1_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
            loss_s0=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s0',
                    connector='loss_s0_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s0',
                    connector='loss_s0_tfeat')),
            loss_s1=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s1',
                    connector='loss_s1_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s1',
                    connector='loss_s1_tfeat')),
            loss_s2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s2',
                    connector='loss_s2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s2',
                    connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s3',
                    connector='loss_s3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s3',
                    connector='loss_s3_tfeat')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
