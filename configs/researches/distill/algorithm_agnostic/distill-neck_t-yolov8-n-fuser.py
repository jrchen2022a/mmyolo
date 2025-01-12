_base_ = './distill-backbone_t-yolov8-n-fuser.py'

model = dict(
    distiller=dict(
        _delete_=True,
        type='ConfigurableDistiller',
        student_recorders=dict(neck=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(neck=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_s2=dict(),
            loss_s3=dict(),
            loss_s4=dict()),
        loss_forward_mappings=dict(
            loss_s2=dict(
                preds_S=dict(from_student=True,  recorder='neck', data_idx=0),
                preds_T=dict(from_student=False, recorder='neck', data_idx=0)),
            loss_s3=dict(
                preds_S=dict(from_student=True,  recorder='neck', data_idx=1),
                preds_T=dict(from_student=False, recorder='neck', data_idx=1)),
            loss_s4=dict(
                preds_S=dict(from_student=True,  recorder='neck', data_idx=2),
                preds_T=dict(from_student=False, recorder='neck', data_idx=2)))))
