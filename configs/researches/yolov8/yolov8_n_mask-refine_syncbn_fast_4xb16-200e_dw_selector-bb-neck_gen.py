_base_ = './yolov8_n_mask-refine_syncbn_fast_4xb16-200e_dw_selector-bb-neck.py'

work_dir = _base_.test_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.project=_base_.wandb_gen_project_name
_base_.visualizer.vis_backends[1].init_kwargs.name='{{fileBasenameNoExtension}}'

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
