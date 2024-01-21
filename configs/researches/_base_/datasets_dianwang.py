# -----data related-----
data_root = 'data/researches/'  # Root path of data
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
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
