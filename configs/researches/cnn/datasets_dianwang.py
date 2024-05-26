_base_ = 'mmdet::_base_/datasets/coco_detection.py'

# dataset settings
data_root = 'data/researches/'
# Path of train annotation file
train_ann_file = 'annotations/trainval.json'
train_data_prefix = 'images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/test.json'
val_data_prefix = 'images/'  # Prefix of val image path

class_name = ('off', 'on', 'aOff', 'aOn', 'bOff', 'bOn', 'cOff', 'cOn', 'away')  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60),(220, 20, 60),(220, 20, 60),(220, 20, 60),(220, 20, 60),(220, 20, 60),(220, 20, 60),(220, 20, 60),(220, 20, 60)]  # 画图时候的颜色，随便设置即可
)

# -----data related-----
# img_scale = (640, 640)  # width, height
# _base_.train_pipeline[2].scale = img_scale
# _base_.test_pipeline[1].scale = img_scale

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + val_ann_file,
    classwise=True,
    metric=['bbox'])
test_evaluator = val_evaluator

