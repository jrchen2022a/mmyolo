_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw.py',
          '../algorithm_agnostic/distill-neck_yolov8-n-fuser_yolov8-n-shufflenet-neck.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='PKDLoss', loss_weight=50),
            loss_s3=dict(type='PKDLoss', loss_weight=50),
            loss_s4=dict(type='PKDLoss', loss_weight=50))))

train_batch_size_per_gpu = 32
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))
