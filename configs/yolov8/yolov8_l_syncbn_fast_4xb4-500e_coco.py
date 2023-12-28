_base_ = './yolov8_l_syncbn_fast_8xb16-500e_coco.py'
train_batch_size_per_gpu = 4
train_num_workers = 2


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))
