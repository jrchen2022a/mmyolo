_base_ = './yolov7_l_syncbn_fast_4xb16-200e_dw.py'

# ===============================Unmodified in most cases====================
num_classes = _base_.num_classes
num_det_layers = _base_.num_det_layers
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
model = dict(
    backbone=dict(
        arch='Tiny', act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    neck=dict(
        is_tiny_version=True,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        block_cfg=dict(
            _delete_=True, type='TinyDownSampleBlock', middle_ratio=0.25),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(in_channels=[128, 256, 512]),
        loss_cls=dict(loss_weight=_base_.loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=_base_.loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

work_dir = './work_dirs/dianwang/yolov7_tiny_syncbn_fast_'+str(_base_.nGPU)+'xb'+str(_base_.train_batch_size_per_gpu)+'-'+str(_base_.max_epochs)+'e_dw/'
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',init_kwargs=dict(_base_.wandb_init_kwargs,name='yolov7_t'))
    ],
    name='visualizer')


