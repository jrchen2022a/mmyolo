# 当你想调试时，_base_引入这个文件，并更改如下_base_
_base_ = './researches/distill/students/yolov8-n-shufflenet-neck_syncbn_fast_4xb16-200e_dw.py'

_base_.train_dataloader.batch_size = 2
_base_.optim_wrapper.optimizer.batch_size_per_gpu = 2
_base_.train_dataloader.num_workers = 1

work_dir = 'work_dirs/temp/'
del _base_.visualizer.vis_backends[1]
