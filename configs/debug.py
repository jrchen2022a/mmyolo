_base_ = 'researches/yolov8/yolov8_n_mask-refine_syncbn_fast_4xb16-200e_dw_selector-bb-neck.py'

_base_.train_dataloader.batch_size = 2
_base_.optim_wrapper.optimizer.batch_size_per_gpu = 2
_base_.train_dataloader.num_workers = 1

work_dir = 'work_dirs/temp/'
del _base_.visualizer.vis_backends[1]
