# _base_ = '../algorithm_agnostic/debug.py'
_base_ = '../algorithm_agnostic/distill-backbone_yolov8-n-fuser_yolov8-n-dscv2_8xb16_200e_dw.py'

work_dir = (_base_.work_dir_root + '/work_dirs/{0}/{{fileBasenameNoExtension}}/'
            .format(_base_.wandb_project_name))
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project=_base_.wandb_project_name, name='{{fileBasenameNoExtension}}'))
    ])

model = dict(
    distiller=dict(
        connectors=dict(
            s2_connector=dict(
                type='MGDConnector',
                student_channels=_base_.stages_output_channels['n'][1],
                teacher_channels=_base_.stages_output_channels['n'][1],
                lambda_mgd=0.65),
            s3_connector=dict(
                type='MGDConnector',
                student_channels=_base_.stages_output_channels['n'][2],
                teacher_channels=_base_.stages_output_channels['n'][2],
                lambda_mgd=0.65),
            s4_connector=dict(
                type='MGDConnector',
                student_channels=_base_.stages_output_channels['n'][3],
                teacher_channels=_base_.stages_output_channels['n'][3],
                lambda_mgd=0.65)),
        distill_losses=dict(
            loss_mgd_s2=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_s3=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_s4=dict(type='MGDLoss', alpha_mgd=0.00002)),
        loss_forward_mappings=dict(
            _delete_=True,
            loss_mgd_s2=dict(
                preds_S=dict(from_student=True,  recorder='stage_s2', data_idx=0, connector='s2_connector'),
                preds_T=dict(from_student=False, recorder='stage_s2', data_idx=0)),
            loss_mgd_s3=dict(
                preds_S=dict(from_student=True,  recorder='stage_s3', data_idx=0, connector='s3_connector'),
                preds_T=dict(from_student=False, recorder='stage_s3', data_idx=0)),
            loss_mgd_s4=dict(
                preds_S=dict(from_student=True,  recorder='stage_s4', data_idx=0, connector='s4_connector'),
                preds_T=dict(from_student=False, recorder='stage_s4', data_idx=0)))))
