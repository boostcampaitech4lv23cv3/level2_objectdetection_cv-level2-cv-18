_base_ = [
    '../../../_module_/models/ssd300.py', 
    #'../../_module_/datasets/coco_detection.py',
    '../../../_module_/schedules/schedule_adam.py', 
    '../../../_module_/default_runtime.py'
]

TYPE_DATASET = 'CocoDataset'
PATH_DATASET = '/opt/ml/dataset'
LIST_CLASSES = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

log_config = dict(
    interval=50,
    hooks=[
         dict(type='TextLoggerHook'),
         dict(type='MMDetWandbHook',
             init_kwargs={'project': 'Trash Detection', "entity": "light-observer", "name": "SSD300"},
         interval=10,
         log_checkpoint=False,
         log_checkpoint_metadata=True,
         num_eval_images=0,
         bbox_score_thr=0.3)
    ])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=TYPE_DATASET,
        ann_file=PATH_DATASET + '/train_split.json',
        img_prefix=PATH_DATASET ,
        classes= LIST_CLASSES,
        pipeline=train_pipeline),
    val=dict(
        type=TYPE_DATASET,
        ann_file=PATH_DATASET + '/val_split.json',
        img_prefix=PATH_DATASET ,
        classes= LIST_CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=TYPE_DATASET,
        ann_file=PATH_DATASET + '/test.json',
        classes= LIST_CLASSES,
        img_prefix=PATH_DATASET ,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')