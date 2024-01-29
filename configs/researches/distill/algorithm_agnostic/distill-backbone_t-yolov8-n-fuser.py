stages_output_channels = {
    'n': [32, 64, 128, 256],
    's': [64, 128, 256, 512],
    'm': [96, 192, 384, 576],
    'l': [128, 256, 512, 512],
    'x': [160, 320, 640, 640]
}

teacher_ckpt = 'checkpoints/yolov8-n-fuser_best-dw.pth'

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    teacher=dict(
        cfg_path='mmyolo::researches/yolov8/yolov8_n_syncbn_fast_4xb16-200e_dw_fuser.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(backbone=dict(type='ModuleOutputs', source='backbone')),
        teacher_recorders=dict(backbone=dict(type='ModuleOutputs', source='backbone')),
        distill_losses=dict(
            loss_s2=dict(),
            loss_s3=dict(),
            loss_s4=dict()),
        loss_forward_mappings=dict(
            loss_s2=dict(
                preds_S=dict(from_student=True,  recorder='backbone', data_idx=0),
                preds_T=dict(from_student=False, recorder='backbone', data_idx=0)),
            loss_s3=dict(
                preds_S=dict(from_student=True,  recorder='backbone', data_idx=1),
                preds_T=dict(from_student=False, recorder='backbone', data_idx=1)),
            loss_s4=dict(
                preds_S=dict(from_student=True,  recorder='backbone', data_idx=2),
                preds_T=dict(from_student=False, recorder='backbone', data_idx=2)))))

find_unused_parameters = True
