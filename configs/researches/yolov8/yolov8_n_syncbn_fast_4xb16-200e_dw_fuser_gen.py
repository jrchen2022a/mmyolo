_base_ = './yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py'

work_dir = ('/home/jrchen/researches/mmyolo_older/test_dirs/{0}/yolov8_n_syncbn_fast_{1}xb{2}-{3}e_dw_fuser/'
            .format(_base_.wandb_standard_gen_project_name, str(_base_.nGPU), str(_base_.train_batch_size_per_gpu), str(_base_.max_epochs)))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=_base_.wandb_standard_gen_project_name, name='yolov8_n_fuser'))
    ])

test_cfg = dict(
    type='GeneralityTestLoop',
    gen_type='standard')
