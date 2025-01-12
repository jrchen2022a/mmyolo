save_work_dir_root = './work_dirs/'
default_scope = 'mmyolo'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

wandb_project_name = 'research_dw_models_exp_v2'
wandb_gen_project_name = 'researches_generality_test_v2'
wandb_standard_gen_project_name = 'researches_generality_test_standard'
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=wandb_project_name))
    ],
    name='visualizer')
work_dir_root = save_work_dir_root + wandb_project_name + '/'
test_dir_root = './test_dirs/'+wandb_gen_project_name+'/'

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# file_client_args = dict(
#         backend='petrel',
#         path_mapping=dict({
#             './data/': 's3://openmmlab/datasets/detection/',
#             'data/': 's3://openmmlab/datasets/detection/'
#         }))
file_client_args = dict(backend='disk')

# Save model checkpoint and validation intervals
save_epoch_intervals = 5
