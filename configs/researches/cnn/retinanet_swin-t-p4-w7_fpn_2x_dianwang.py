_base_ = [
    'mmdet::_base_/models/retinanet_r50_fpn.py',
    'datasets_dianwang.py',
    'schedule_2x_dianwang.py',
    'runtime_dianwang.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5))

# optimizer

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name='{{fileBasenameNoExtension}}'


_base_.model.bbox_head.num_classes = _base_.num_classes
# train_dataloader = dict(batch_size=1)
#
# val_dataloader = dict(batch_size=1)
# test_dataloader = val_dataloader
optim_wrapper = dict(optimizer=dict(lr=0.002))
# auto_scale_lr = dict(enable=False, base_batczh_size=4)