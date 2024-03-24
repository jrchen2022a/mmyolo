_base_ = ['./distill-neck_t-yolov8-n-fuser.py',
          '../students/yolov8-n-lite.py']

student = _base_.architecture
# student.neck.init_cfg = dict(
#     type='Pretrained', prefix='neck.', checkpoint=teacher_ckpt)
# student.bbox_head.init_cfg = dict(
#     type='Pretrained', prefix='bbox_head.', checkpoint=teacher_ckpt)
model = dict(
    architecture=student)

del student
del _base_.architecture
