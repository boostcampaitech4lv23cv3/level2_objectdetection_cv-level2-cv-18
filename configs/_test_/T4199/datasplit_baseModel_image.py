_base_ = [
    'datasets/split_trash_datasets_aug.py',
    'model/faster_rcnn_r50_fpn_resnext.py',
    'schedules/schedule_1x.py',
    '4199_runtime_wandb_image.py'
]
