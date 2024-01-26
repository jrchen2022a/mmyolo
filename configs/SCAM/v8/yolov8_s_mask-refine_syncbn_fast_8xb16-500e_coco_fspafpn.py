_base_ = './yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
# _base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'
train_batch_size_per_gpu = 32
_base_.train_dataloader.batch_size=train_batch_size_per_gpu
_base_.optim_wrapper.optimizer.batch_size_per_gpu=train_batch_size_per_gpu

model = dict(
    neck=dict(
        type='YOLOv8FSPAFPN'))