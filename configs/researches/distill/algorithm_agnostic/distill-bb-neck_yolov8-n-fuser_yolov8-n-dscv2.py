_base_ = ['./distill-backbone_yolov8-n-fuser_yolov8-n-dscv2.py']

model = dict(
    distiller=dict(
        student_recorders=dict(
            neck_s2=dict(type='ModuleOutputs', source='neck.out_layers.0'),
            neck_s3=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            neck_s4=dict(type='ModuleOutputs', source='neck.out_layers.2')),
        teacher_recorders=dict(
            neck_s2=dict(type='ModuleOutputs', source='neck.out_layers.0'),
            neck_s3=dict(type='ModuleOutputs', source='neck.out_layers.1'),
            neck_s4=dict(type='ModuleOutputs', source='neck.out_layers.2')),
        distill_losses=dict(
            loss_ns2=dict(),
            loss_ns3=dict(),
            loss_ns4=dict()),
        loss_forward_mappings=dict(
            loss_ns2=dict(
                preds_S=dict(from_student=True,  recorder='neck_s2'),
                preds_T=dict(from_student=False, recorder='neck_s2')),
            loss_ns3=dict(
                preds_S=dict(from_student=True,  recorder='neck_s3'),
                preds_T=dict(from_student=False, recorder='neck_s3')),
            loss_ns4=dict(
                preds_S=dict(from_student=True,  recorder='neck_s4'),
                preds_T=dict(from_student=False, recorder='neck_s4')))))
