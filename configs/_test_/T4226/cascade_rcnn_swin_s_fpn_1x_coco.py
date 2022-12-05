_base_ = [
    '../../_module_/models/cascade_rcnn_swin_s_fpn.py',
    '../../_module_/datasets/coco_detection.py',
    '../../_module_/schedules/schedule_1x.py',
    '../../_module_/4226_base_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8
)
auto_scale_lr = dict(enable=True, base_batch_size=32)

max_epochs = 64
num_last_epochs = 15
interval = 10
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=max_epochs)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                 'project': 'Trash Detection',
                 "entity": "light-observer",
                 "name": "Cascade RCNN Swin-s"
             },
             interval=10,
             log_checkpoint=False,
             log_checkpoint_metadata=True,
             num_eval_images=100,
             bbox_score_thr=0.3)
    ])