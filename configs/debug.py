_base_ = 'researches/distill/spatialchannelkd/sckd-bb_yolov8-n-fuser_yolov8-n-lite_mask-refine_syncbn_fast_4xb16_200e_dw.py'

_base_.train_dataloader.batch_size = 2
_base_.optim_wrapper.optimizer.batch_size_per_gpu = 2
_base_.train_dataloader.num_workers = 1

work_dir = 'work_dirs/temp/'
del _base_.visualizer.vis_backends[1]

