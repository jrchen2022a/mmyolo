_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'
train_batch_size_per_gpu = 8
train_num_workers = 4


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))
