train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
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
        milestones=[26, 29],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
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
load_from = '../checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
resume = False
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_size_divisor=32)
model = dict(
    type='YOLOV3',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=1,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        ann_file='330/test/coco_instances_image.json',
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data_hdd/tao/Datasets/330/val/coco_instances_image.json',
    metric='bbox',
    backend_args=None,
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data_hdd/tao/Datasets/330/test/coco_instances_image.json',
    metric='bbox',
    backend_args=None,
    format_only=False)
custom_imports = dict(imports=['new_pipeline'], allow_failed_imports=False)
img_bands_index = dict(bands_index=[5, 3, 1])
img_norm_method = dict(method='min-max')
metainfo = dict(classes=('tree', ), palette=[(220, 20, 60)])
checkpoint = dict(type='CheckpointHook', interval=5)
valid_pipeline = [
    dict(type='LoadRSImageFromFile', bands_index=[5, 3, 1]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Norma', method='min-max'),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
seed = 42
gpu_ids = range(0, 2)
device = 'cuda'
work_dir = './log/yolov3'
launcher = 'pytorch'
