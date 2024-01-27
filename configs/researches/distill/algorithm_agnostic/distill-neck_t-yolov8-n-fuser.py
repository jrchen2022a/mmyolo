_base_ = './distill-backbone_t-yolov8-n-fuser.py'

model = dict(
    distiller=dict(
        _delete_=True,
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck_s2=dict(type='ModuleOutputs', source='neck.out_layers.0'),
            neck_s3=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            neck_s4=dict(type='ModuleOutputs', source='neck.out_layers.2')),
        teacher_recorders=dict(
            neck_s2=dict(type='ModuleOutputs', source='neck.out_layers.0'),
            neck_s3=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            neck_s4=dict(type='ModuleOutputs', source='neck.out_layers.2')),
        distill_losses=dict(
            loss_s2=dict(),
            loss_s3=dict(),
            loss_s4=dict()),
        loss_forward_mappings=dict(
            loss_s2=dict(
                preds_S=dict(from_student=True,  recorder='neck_s2'),
                preds_T=dict(from_student=False, recorder='neck_s2')),
            loss_s3=dict(
                preds_S=dict(from_student=True,  recorder='neck_s3'),
                preds_T=dict(from_student=False, recorder='neck_s3')),
            loss_s4=dict(
                preds_S=dict(from_student=True,  recorder='neck_s4'),
                preds_T=dict(from_student=False, recorder='neck_s4')))))
