_base_ = './yolov8_n_syncbn_fast_4xb16-200e_dw.py'

work_dir = (_base_.work_dir_root+'/test_dirs/{0}/yolov8_n_syncbn_fast_4xb16-200e_dw/'
            .format(_base_.wandb_standard_gen_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project=_base_.wandb_gen_project_name, name='yolov8_n'))
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
