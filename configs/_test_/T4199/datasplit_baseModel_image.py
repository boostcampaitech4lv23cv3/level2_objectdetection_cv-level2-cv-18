_base_ = [
    'datasets/split_trash_datasets_800_640.py',
    'model/faster_rcnn_r50_fpn.py',
    'schedules/schedule_1x.py',
    '4199_runtime_wandb_image.py'
]
