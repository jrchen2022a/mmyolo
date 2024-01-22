# 当你想调试时，_base_引入这个文件，将batch_size调成2

train_batch_size_per_gpu = 2
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))