_base_ = [
    '../../_module_/models/faster_rcnn_r50_fpn.py',
    '../../_module_/datasets/coco_modified_detection.py',
    '../../_module_/schedules/schedule_2x.py',
    '../../_module_/4226_base_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.3)
        )
    )
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4
)

auto_scale_lr = dict(enable=True, base_batch_size=32)
