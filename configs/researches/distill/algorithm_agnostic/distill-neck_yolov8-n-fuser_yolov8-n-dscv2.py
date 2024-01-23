_base_ = ['./distill-backbone_yolov8-n-fuser_yolov8-n-dscv2.py']

model = dict(
    distiller=dict(
        _delete_=True
        ))
