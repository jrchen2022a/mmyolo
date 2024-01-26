_base_ = './yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_fuser-bb-neck.py'

work_dir = _base_.work_dir_root+'/work_dirs/coco/{{fileBasenameNoExtension}}/'

train_batch_size_per_gpu = 32
_base_.train_dataloader.batch_size=train_batch_size_per_gpu
_base_.optim_wrapper.optimizer.batch_size_per_gpu=train_batch_size_per_gpu

