_base_ = [
    '../../../_module_/models/retinanet_r50_fpn.py',
    '../../../_module_/datasets/coco_detection.py',
    '../../../_module_/schedules/schedule_1x.py', 
    '../../../_module_/default_runtime.py'
]
WANDB_RUN_NAME = 'retinanet_r18_fpn_1x'
# optimizer
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

log_config = dict(
    interval=50,
    hooks=[
         dict(type='TextLoggerHook'),
         dict(type='MMDetWandbHook',
             init_kwargs={'project': 'Trash Detection', "entity": "light-observer", "name": WANDB_RUN_NAME},
         interval=10,
         log_checkpoint=False,
         log_checkpoint_metadata=True,
         num_eval_images=0,
         bbox_score_thr=0.3)
    ])