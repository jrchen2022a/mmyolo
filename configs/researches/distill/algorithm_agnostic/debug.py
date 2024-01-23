# 当你想调试时，_base_引入这个文件，并更改如下_base_
_base_ = '../../../yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py'

train_batch_size_per_gpu = 2
_base_.train_dataloader.batch_size=train_batch_size_per_gpu
_base_.optim_wrapper.optimizer.batch_size_per_gpu=train_batch_size_per_gpu
work_dir = 'work_dirs/temp/'
_base_.visualizer.vis_backends=[dict(type='LocalVisBackend')]
