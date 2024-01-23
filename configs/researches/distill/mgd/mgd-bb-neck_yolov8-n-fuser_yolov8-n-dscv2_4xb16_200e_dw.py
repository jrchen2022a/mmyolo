# _base_ = '../algorithm_agnostic/debug.py'
_base_ = ['../algorithm_agnostic/distill-dw_schedule_8xb16_200e_dw.py',
          '../algorithm_agnostic/distill-bb-neck_yolov8-n-fuser_yolov8-n-dscv2.py']

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

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
                lambda_mgd=0.65),
            ns2_connector=dict(
                type='MGDConnector',
                student_channels=_base_.stages_output_channels['n'][1],
                teacher_channels=_base_.stages_output_channels['n'][1],
                lambda_mgd=0.65),
            ns3_connector=dict(
                type='MGDConnector',
                student_channels=_base_.stages_output_channels['n'][2],
                teacher_channels=_base_.stages_output_channels['n'][2],
                lambda_mgd=0.65),
            ns4_connector=dict(
                type='MGDConnector',
                student_channels=_base_.stages_output_channels['n'][3],
                teacher_channels=_base_.stages_output_channels['n'][3],
                lambda_mgd=0.65)),
        distill_losses=dict(
            loss_s2=dict(type='MGDLoss', alpha_mgd=0.002),
            loss_s3=dict(type='MGDLoss', alpha_mgd=0.004),
            loss_s4=dict(type='MGDLoss', alpha_mgd=0.006),
            loss_ns2=dict(type='MGDLoss', alpha_mgd=0.002),
            loss_ns3=dict(type='MGDLoss', alpha_mgd=0.004),
            loss_ns4=dict(type='MGDLoss', alpha_mgd=0.006)),
        loss_forward_mappings=dict(
            loss_s2=dict(preds_S=dict(connector='s2_connector')),
            loss_s3=dict(preds_S=dict(connector='s3_connector')),
            loss_s4=dict(preds_S=dict(connector='s4_connector')),
            loss_ns2=dict(preds_S=dict(connector='ns2_connector')),
            loss_ns3=dict(preds_S=dict(connector='ns3_connector')),
            loss_ns4=dict(preds_S=dict(connector='ns4_connector')))))

del _base_.train_cfg['dynamic_intervals']
del _base_.custom_hooks[1]
del _base_.val_interval_stage2
del _base_.train_pipeline
del _base_.train_pipeline_stage2

