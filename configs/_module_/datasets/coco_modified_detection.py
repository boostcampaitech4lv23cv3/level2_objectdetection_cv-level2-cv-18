# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
<<<<<<< HEAD
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
=======
    dict(type='LoadAnnotations', with_bbox=True),
>>>>>>> feat/faster-rcnn
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
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
        img_scale=(512, 512),
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

classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
<<<<<<< HEAD
        ann_file=data_root + '/train.json',
=======
        ann_file=data_root + '/train_eda_dropminmax.json',
>>>>>>> feat/faster-rcnn
        img_prefix=data_root ,
        classes= classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
<<<<<<< HEAD
        ann_file=data_root + '/train.json',
=======
        ann_file=data_root + '/train_eda_dropminmax.json',
>>>>>>> feat/faster-rcnn
        img_prefix=data_root ,
        classes= classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/test.json',
        classes= classes,
        img_prefix=data_root ,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
