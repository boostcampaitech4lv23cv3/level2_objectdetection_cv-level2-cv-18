checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
<<<<<<< HEAD
         dict(type='TextLoggerHook'),
         # dict(type='TensorboardLoggerHook'),
         dict(type='MMDetWandbHook',
             init_kwargs={'project': 'Trash Detection', "entity": "light-observer"},
         interval=10,
         log_checkpoint=False,
         log_checkpoint_metadata=True,
         num_eval_images=0,
         bbox_score_thr=0.3)
=======
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
>>>>>>> 7c92242a3db632e32d4a2b5cf2c1b3e17cbbcecc
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
