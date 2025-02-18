dataset_type = 'CocoDataset'
data_root = '/data_hdd/tao/Datasets/'
backend_args = None
train_pipeline = [
    dict(type='LoadRSImageFromFile', bands_index=[5, 3, 1]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Norma', method='min-max'),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Norma', method='min-max'),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='/data_hdd/tao/Datasets/',
        ann_file='330/train/coco_instances_image.json',
        data_prefix=dict(img='330/train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadRSImageFromFile', bands_index=[5, 3, 1]),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Norma', method='min-max'),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        backend_args=None,
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/data_hdd/tao/Datasets/',
        ann_file='330/val/coco_instances_image.json',
        data_prefix=dict(img='330/val/images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadRSImageFromFile', bands_index=[5, 3, 1]),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Norma', method='min-max'),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/data_hdd/tao/Datasets/',
        ann_file='330/val/coco_instances_image.json',
        data_prefix=dict(img='330/val/images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromNDArray'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Norma', method='min-max'),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data_hdd/tao/Datasets/330/val/coco_instances_image.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data_hdd/tao/Datasets/330/val/coco_instances_image.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=30),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '../checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
resume = False
model = dict(
    type='DETR',
    num_queries=300,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)))),
    decoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type='DETRHead',
        num_classes=1,
        embed_dims=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
max_epochs = 300
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[296, 299],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=16, enable=False)
img_norm_cfg = None
custom_imports = dict(imports=['new_pipeline'], allow_failed_imports=False)
img_bands_index = dict(bands_index=[5, 3, 1])
img_norm_method = dict(method='min-max')
metainfo = dict(classes=('tree', ), palette=[(220, 20, 60)])
seed = 42
gpu_ids = range(0, 2)
device = 'cuda'
work_dir = './log/detr_300'
launcher = 'pytorch'
