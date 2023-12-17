_base_ = './yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py'

work_dir = '/home/jrchen/researches/mmyolo_older/test_dirs/dianwang/yolov8_n_syncbn_fast_' + str(_base_.nGPU) + 'xb' + str(
    _base_.train_batch_size_per_gpu) + '-' + str(_base_.max_epochs) + 'e_dw_fuser/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=_base_.wandb_gen_project_name, name='yolov8_n_fuser'))
    ])

test_cfg = dict(
    type='GeneralityTestLoop',
    corruptions=['guassian_noise',
                 'shot_noise',
                 'impulse_noise',
                 'defocus_blur',
                 'frosted_glass_blur',
                 'motion_blur',
                 'zoom_blur',
                 'snow',
                 'rain',
                 'fog',
                 'brightness',
                 'contrast',
                 'elastic',
                 'pixelate',
                 'jpeg'],
    severities=[1, 2, 3, 4, 5])
