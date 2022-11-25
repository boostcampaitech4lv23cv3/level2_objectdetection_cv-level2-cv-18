_base_ = './retinanet_r50_fpn_1x.py'

WANDB_RUN_NAME = 'retinanet_r50_fpn_2x'

# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

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