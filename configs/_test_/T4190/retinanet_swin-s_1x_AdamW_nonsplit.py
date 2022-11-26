_base_ = [
    '../../_module_/models/retinanet_r50_fpn.py',
    '../../_module_/datasets/coco_detection_aug_640.py',
    '../../_module_/schedules/schedule_adam.py', '../../_module_/4190_runtime_wandb.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5))

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=0.0005,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
log_config = dict(
    interval=50,
    hooks=[
         dict(type='TextLoggerHook'),
         dict(type='MMDetWandbHook',
             init_kwargs={'project': 'retinanet_swin_s', "entity": "light-observer"},
         interval=50,
         log_checkpoint=False,
         log_checkpoint_metadata=True,
         num_eval_images=10,
         bbox_score_thr=0.3)
    ])
checkpoint_config = dict(interval=10)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=1,
    metric='bbox')
auto_scale_lr = dict(enable=False, base_batch_size=64)
runner = dict(type='EpochBasedRunner', max_epochs=50)