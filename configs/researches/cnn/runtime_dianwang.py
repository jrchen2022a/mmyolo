_base_ = 'mmdet::_base_/default_runtime.py'

save_work_dir_root = './work_dirs/'
wandb_project_name = 'research_dw_models_exp_v2'
wandb_gen_project_name = 'researches_generality_test_v2'
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=wandb_project_name))
    ],
    name='visualizer')
work_dir_root = save_work_dir_root + wandb_project_name + '/'
test_dir_root = './test_dirs/'+wandb_gen_project_name+'/'

