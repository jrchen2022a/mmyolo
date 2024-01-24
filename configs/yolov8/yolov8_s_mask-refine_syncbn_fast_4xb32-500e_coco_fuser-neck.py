_base_ = './yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_fuser-neck.py'

work_dir = _base_.work_dir_root+'/work_dirs/coco/{{fileBasenameNoExtension}}/'

train_batch_size_per_gpu = 32
train_num_workers = 16
_base_.train_dataloader.batch_size=train_batch_size_per_gpu
_base_.train_dataloader.num_workers=train_num_workers
_base_.optim_wrapper.optimizer.batch_size_per_gpu=train_batch_size_per_gpu

