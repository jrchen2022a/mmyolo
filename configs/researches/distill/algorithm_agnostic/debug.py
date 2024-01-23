# 当你想调试时，_base_引入这个文件，并更改如下_base_
_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw.py',
          '../algorithm_agnostic/distill-bb-neck_yolov8-n-fuser_yolov8-n-dscv2.py']

_base_.train_dataloader.batch_size = 2
_base_.optim_wrapper.optimizer.batch_size_per_gpu = 2
_base_.train_dataloader.num_workers = 1

work_dir = 'work_dirs/temp/'
_base_.visualizer.vis_backends=[dict(type='LocalVisBackend')]
