# _base_ = '../algorithm_agnostic/debug.py'
_base_ = '../../../yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py'

train_batch_size_per_gpu = 64
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = _base_.work_dir_root+'/work_dirs/coco/{{fileBasenameNoExtension}}/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='mgd-backbone_yolov8-l-fuser-mr_yolov8-n'))
    ])

stages_output_channels = {
    'n': [32, 64, 128, 256],
    's': [64, 128, 256, 512],
    'm': [96, 192, 384, 576],
    'l': [128, 256, 512, 512],
    'x': [160, 320, 640, 640]
}

teacher_ckpt = 'checkpoints/yolov8-l-fuser_best-coco.pth'
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmyolo::yolov8/yolov8_l_mask-refine_syncbn_fast_4xb32-500e_coco_fuser.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            stage_s2=dict(type='ModuleInputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleInputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleInputs', source='neck.reduce_layers.2')),
        teacher_recorders=dict(
            stage_s2=dict(type='ModuleInputs', source='neck.reduce_layers.0'),
            stage_s3=dict(type='ModuleInputs', source='neck.reduce_layers.1'),
            stage_s4=dict(type='ModuleInputs', source='neck.reduce_layers.2')),
        connectors=dict(
            s2_connector=dict(
                type='MGDConnector',
                student_channels=stages_output_channels['n'][1],
                teacher_channels=stages_output_channels['l'][1],
                lambda_mgd=0.65),
            s3_connector=dict(
                type='MGDConnector',
                student_channels=stages_output_channels['n'][2],
                teacher_channels=stages_output_channels['l'][2],
                lambda_mgd=0.65),
            s4_connector=dict(
                type='MGDConnector',
                student_channels=stages_output_channels['n'][3],
                teacher_channels=stages_output_channels['l'][3],
                lambda_mgd=0.65)),
        distill_losses=dict(
            loss_mgd_s2=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_s3=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_s4=dict(type='MGDLoss', alpha_mgd=0.00002)),
        loss_forward_mappings=dict(
            loss_mgd_s2=dict(
                preds_S=dict(from_student=True, recorder='stage_s2', data_idx=0, connector='s2_connector'),
                preds_T=dict(from_student=False, recorder='stage_s2', data_idx=0)),
            loss_mgd_s3=dict(
                preds_S=dict(from_student=True, recorder='stage_s3', data_idx=0, connector='s3_connector'),
                preds_T=dict(from_student=False, recorder='stage_s3', data_idx=0)),
            loss_mgd_s4=dict(
                preds_S=dict(from_student=True, recorder='stage_s4', data_idx=0, connector='s4_connector'),
                preds_T=dict(from_student=False, recorder='stage_s4', data_idx=0)))))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
