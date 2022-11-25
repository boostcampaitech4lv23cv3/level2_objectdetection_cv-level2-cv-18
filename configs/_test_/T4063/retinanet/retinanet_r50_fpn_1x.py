_base_ = [
    '../../../_module_/models/retinanet_r50_fpn.py',
    '../../../_module_/datasets/coco_detection.py',
    '../../../_module_/schedules/schedule_1x.py', 
    '../../../_module_/default_runtime.py'
]
WANDB_RUN_NAME = 'retinanet_r50_fpn_1x'
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

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