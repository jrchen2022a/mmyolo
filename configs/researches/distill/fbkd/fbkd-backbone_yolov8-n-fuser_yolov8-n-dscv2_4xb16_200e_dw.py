_base_ = '../algorithm_agnostic/distill-backbone_yolov8-n-fuser_yolov8-n-dscv2_8xb16_200e_dw.py'

work_dir = _base_.work_dir_root+'{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends[1].init_kwargs.name = '{{fileBasenameNoExtension}}'

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss'),
            loss_s4=dict(type='FBKDLoss')),
        connectors=dict(
            loss_s2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=_base_.stages_output_channels['n'][1],
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=_base_.stages_output_channels['n'][1],
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=_base_.stages_output_channels['n'][2],
                mode='dot_product',
                sub_sample=True),
            loss_s3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=_base_.stages_output_channels['n'][2],
                mode='dot_product',
                sub_sample=True),
            loss_s4_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=_base_.stages_output_channels['n'][3],
                mode='dot_product',
                sub_sample=True),
            loss_s4_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=_base_.stages_output_channels['n'][3],
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
            loss_s2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s2',
                    connector='loss_s2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s2',
                    connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s3',
                    connector='loss_s3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s3',
                    connector='loss_s3_tfeat')),
            loss_s4=dict(
                s_input=dict(
                    from_student=True,
                    recorder='stage_s4',
                    connector='loss_s4_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='stage_s4',
                    connector='loss_s4_tfeat')))))
