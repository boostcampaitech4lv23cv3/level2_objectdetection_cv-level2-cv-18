_base_ = [
    '../../_module_/models/retinanet_swin_t_fpn.py',
    '../../_module_/datasets/coco_detection.py',
    '../../_module_/schedules/schedule_1x.py',
    '../../_module_/4226_base_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
auto_scale_lr = dict(enable=True, base_batch_size=16)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8
)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                 'project': 'Trash Detection',
                 "entity": "light-observer",
                 "name": "Cascade RCNN FocalNet"
             },
             interval=5,
             log_checkpoint=False,
             log_checkpoint_metadata=True,
             num_eval_images=100,
             bbox_score_thr=0.3)
    ])