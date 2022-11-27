_base_ = [
    '../../../_module_/amp/mixed_precision_fixed.py',
    './retinanet_r50_fpn_1x.py',
]

WANDB_RUN_NAME = 'retinanet_r101_fpn_1x'
MAX_EPOCHS = 50

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0004,paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=MAX_EPOCHS)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])

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