_base_ = ['../../configs/dn_detr/dn_detr_r50_dc5_8x2_12e_coco.py']
model = dict(
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
    bbox_head=dict(
        num_classes=1))
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

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# optimizer
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
# learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
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
auto_scale_lr = dict(enable=False, base_batch_size=16)
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
load_from = '../checkpoints/dn_detr_r50_50ep.pth'
# load_from = 'checkpoints/detr-r101-2c7b67e5.pth'
resume = False
img_norm_cfg = None
custom_imports = dict(imports=['new_pipeline'], allow_failed_imports=False)
img_bands_index = dict(bands_index=[5, 3, 1])
img_norm_method = dict(method='min-max')
metainfo = dict(classes=('tree', ), palette=[(220, 20, 60)])
seed = 42
gpu_ids = range(0, 2)
device = 'cuda'
work_dir = './log/dn_detr'
launcher = 'pytorch'
