_base_ = './yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py'

work_dir = _base_.work_dir_root+'/work_dirs/coco/{{fileBasenameNoExtension}}/'
_base_.visualizer.vis_backends.append(
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='research_coco',
                          name='yolov8_m_fuser-neck_mr')))

model = dict(
    neck=dict(
        type='YOLOv8SelectorPAFPN',
        num_selectors=3))
