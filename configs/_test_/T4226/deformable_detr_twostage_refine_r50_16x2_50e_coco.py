_base_ = 'deformable_detr_refine_r50_16x2_50e_coco.py'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
model = dict(bbox_head=dict(as_two_stage=True))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                 'project': 'Trash Detection',
                 "entity": "light-observer",
                 "name": "Deformable Detr Tho-stage"
             },
             interval=10,
             log_checkpoint=False,
             log_checkpoint_metadata=True,
             num_eval_images=100,
             bbox_score_thr=0.3)
    ])