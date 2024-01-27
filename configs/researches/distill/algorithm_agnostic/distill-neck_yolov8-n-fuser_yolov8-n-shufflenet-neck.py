_base_ = ['./distill-neck_t-yolov8-n-fuser.py',
          '../students/yolov8-n-shufflenet-neck.py']

student = _base_.architecture
student.bbox_head.init_cfg = dict(
    type='Pretrained', prefix='bbox_head.', checkpoint=_base_.teacher_ckpt)
model = dict(
    architecture=student)

del student
del _base_.architecture
