_base_ = ['../configs/pvt/retinanet_pvt-s_fpn_1x_coco.py']

custom_imports = dict(imports=['new_pipeline'], allow_failed_imports=False)
img_bands_index = dict(bands_index=[5, 3, 1])
img_norm_method = dict(method='min-max')
metainfo = dict(classes=('tree', ), palette=[(220, 20, 60)])
checkpoint=dict(type='CheckpointHook', interval=5)

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_size_divisor=32)

model = dict(
    data_preprocessor=data_preprocessor,
    bbox_head=dict(
        num_classes=1))

train_pipeline = [
    dict(type='LoadRSImageFromFile', **img_bands_index),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Norma', **img_norm_method),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')]

valid_pipeline = [
    dict(type='LoadRSImageFromFile', **img_bands_index),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Norma', **img_norm_method),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]

test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Norma', **img_norm_method),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]


train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root='/data_hdd/tao/Datasets/',
        ann_file='315/train/coco_instances_image.json',
        data_prefix=dict(img='315/train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root='/data_hdd/tao/Datasets/',
        ann_file='315/val/coco_instances_image.json',
        data_prefix=dict(img='315/val/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=valid_pipeline,
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root='/data_hdd/tao/Datasets/',
        ann_file='315/val/coco_instances_image.json',
        data_prefix=dict(img='315/val/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        metainfo=dict(classes=('tree', ), palette=[(220, 20, 60)])))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data_hdd/tao/Datasets/315/val/coco_instances_image.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/data_hdd/tao/Datasets/315/val/coco_instances_image.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
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
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001))
default_hooks = dict(
    checkpoint=checkpoint)

load_from = 'checkpoints/retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth'
seed = 42
gpu_ids = range(0, 2)
device = 'cuda'
work_dir = './log/retinanet_pvt'
launcher = 'pytorch'





