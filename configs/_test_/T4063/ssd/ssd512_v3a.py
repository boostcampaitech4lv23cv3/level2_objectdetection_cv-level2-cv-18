INPUT_SIZE = 512
NUM_CLASSES = 10
DATASET_TYPE = 'CocoDataset'
DATA_ROOT = '/opt/ml/dataset/'
CLASS_LIST = [
    'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
    'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
WANDB_RUN_NAME = 'SSD512v3a'
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
AMP = 'dynamic' # dynamic or 512.
EPOCH = 10
CHECKPOINT_SAVE_INTERVAL = 5
CHECKPOINT_LOAD_PATH = './work_dirs/ssd512_base/epoch_24.pth'
LOG_INTERVAL = 10
INVALID_LOSS_CHECK_INTERVAL = 500
EVALUATION_INTERVAL = 1
LR = 1e-4
TRAIN_JSON = 'train_eda_dropminmax.json'
VAL_JSON = 'train_eda_dropminmax.json'
TEST_JSON = 'test.json'
WARMUP_ITERS = 10
WARMUP_RATIO = 0.001


# Model
load_from = CHECKPOINT_LOAD_PATH
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34)),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        l2_norm_scale=20,
        last_kernel_size=4),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        num_classes=NUM_CLASSES,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=512,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))

cudnn_benchmark = True

# Data Pipeline
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# Data Loader
data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=DATA_ROOT + TRAIN_JSON,
        img_prefix=DATA_ROOT,
        classes=CLASS_LIST,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Expand',
                mean=[123.675, 116.28, 103.53],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(INPUT_SIZE, INPUT_SIZE), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        samples_per_gpu=VAL_BATCH_SIZE,
        ann_file=DATA_ROOT + VAL_JSON,
        img_prefix=DATA_ROOT,
        classes=CLASS_LIST,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(INPUT_SIZE, INPUT_SIZE),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=DATA_ROOT + TEST_JSON,
        classes=CLASS_LIST,
        img_prefix=DATA_ROOT,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(INPUT_SIZE, INPUT_SIZE),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

# evaluation
evaluation = dict(interval=EVALUATION_INTERVAL, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=LR, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=WARMUP_ITERS,
    warmup_ratio=WARMUP_RATIO,
    step=[16, 22])

# Runner
runner = dict(type='EpochBasedRunner', max_epochs=EPOCH)

# checkpoint
checkpoint_config = dict(interval=CHECKPOINT_SAVE_INTERVAL)

# wandb Log
log_config = dict(
    interval=LOG_INTERVAL,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                project='Trash Detection',
                entity='light-observer',
                name=WANDB_RUN_NAME),
            interval=10,
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            num_eval_images=0,
            bbox_score_thr=0.3)
    ])

# custom hook
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=INVALID_LOSS_CHECK_INTERVAL, priority='VERY_LOW')
]

# AMP
fp16 = dict(loss_scale=AMP)

# etc
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=8)
auto_resume = False
gpu_ids = [0]