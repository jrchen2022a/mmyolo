# 当你想调试时，_base_引入这个文件，并更改如下_base_
_base_ = './v8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_fspafpn.py'

_base_.train_dataloader.batch_size = 2
_base_.optim_wrapper.optimizer.batch_size_per_gpu = 2
_base_.train_dataloader.num_workers = 1

work_dir = 'work_dirs/temp/'
# del _base_.visualizer.vis_backends[1]
