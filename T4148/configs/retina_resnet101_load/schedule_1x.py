# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

#learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[15, 19]
    )


# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0/10,
#     min_lr_ration=1e-5
#     #step=[23, 27, 45, 50, 57]
#     )

runner = dict(type='EpochBasedRunner', max_epochs=20)
