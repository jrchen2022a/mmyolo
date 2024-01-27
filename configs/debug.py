# 当你想调试时，_base_引入这个文件，并更改如下_base_
_base_ = './researches/distill/pkd/pkd-neck_yolov8-n-fuser_yolov8-n-shufflenet-neck_4xb16_200e_dw-uc-50-hf.py'

_base_.train_dataloader.batch_size = 2
_base_.optim_wrapper.optimizer.batch_size_per_gpu = 2
_base_.train_dataloader.num_workers = 1

work_dir = 'work_dirs/temp/'
del _base_.visualizer.vis_backends[1]
