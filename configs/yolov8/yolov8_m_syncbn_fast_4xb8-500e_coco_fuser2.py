# 占用10G
_base_ = './yolov8_m_syncbn_fast_8xb16-500e_coco.py'
base_lr = 0.0025
optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr))
train_batch_size_per_gpu = 8
train_num_workers = 4

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

work_dir = './work_dirs/coco/yolov8_m_syncbn_fast_4xb8-500e_coco_fuser2/'
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(project='research_coco', name='yolov8_m_fuser2'))
    ])
# work_dir = '/home/jrchen/researches/mmyolo_older/work_dirs/temp/'
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend')     ])

model = dict(
    backbone=dict(
        type='YOLOv8SelectorCSPDarknet',
        num_selectors=2,
        selector_type='SelectorCSPLayerWithTwoConv'))