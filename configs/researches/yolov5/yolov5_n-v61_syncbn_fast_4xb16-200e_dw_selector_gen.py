_base_ = './yolov5_n-v61_syncbn_fast_4xb16-200e_dw_selector.py'

work_dir = '/home/jrchen/researches/mmyolo_older/test_dirs/dianwang/yolov5_n-v61_syncbn_fast_' + str(_base_.nGPU) + 'xb' + str(
    _base_.train_batch_size_per_gpu) + '-' + str(_base_.max_epochs) + 'e_dw_selector_gen/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='researches_generality_test', name='yolov5_n_selector'))
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
