# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------

double_dataset_type = 'ssda_cls_double_dataset'
triple_dataset_type = 'ssda_cls_triple_dataset'
single_dataset_type = 'ssda_cls_dataset'
source_domain = 'officehome_Real'
target_domain = 'officehome_Art'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

rand_range_aug = dict(
    type='RandRangeAug',
    num_policies=2,
    magnitude_level=10,
    policies=[
        dict(type='AutoContrast'),
        dict(type='Brightness', magnitude_key='magnitude', magnitude_range=[0.1, 1.9], prob=0.5),
        dict(type='ColorTransform', magnitude_key='magnitude', magnitude_range=[0.1, 1.9], prob=0.5),
        dict(type='Contrast', magnitude_key='magnitude', magnitude_range=[0.1, 1.9], prob=0.5),
        dict(type='Equalize'),
        dict(type='Identity'),
        dict(type='Posterize', magnitude_key='bits', magnitude_range=[4, 8], prob=0.5),
        dict(type='Rotate', magnitude_key='angle', magnitude_range=[-30, 30], prob=0.5),
        dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=[0.1, 1.9], prob=0.5),
        dict(type='Shear', magnitude_key='magnitude', magnitude_range=[0, 0.3], direction='horizontal', prob=0.5),
        dict(type='Shear', magnitude_key='magnitude', magnitude_range=[0, 0.3], direction='vertical', prob=0.5),
        dict(type='Solarize', magnitude_key='thr', magnitude_range=[0, 256], prob=0.5),
        dict(type='Translate', magnitude_key='magnitude', magnitude_range=[-0.3, 0.3], direction='horizontal',
             prob=0.5),
        dict(type='Translate', magnitude_key='magnitude', magnitude_range=[-0.3, 0.3], direction='vertical',
             prob=0.5),
    ]
)

train_pipelines = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomCrop', size=224),
    rand_range_aug,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

weak_pipelines = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

train_pipelines2 = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    rand_range_aug,
    dict(type='Cutout', shape=16, ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipelines = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

source_datasets = dict(
    type=double_dataset_type,
    name=source_domain,
    split='labeled_source',
    pipeline=weak_pipelines,
    pipeline2=train_pipelines,
    min_len=60000,
    builder=dict(
        samples_per_gpu=12,
    )
)

target_labeled_datasets = dict(
    type=double_dataset_type,
    name=target_domain,
    split='labeled_target',
    pipeline=weak_pipelines,
    pipeline2=train_pipelines,
    shot=3,
    min_len=60000,
    builder=dict(
        samples_per_gpu=12,
    )
)

target_unlabeled_datasets = dict(
    type=triple_dataset_type,
    name=target_domain,
    split='unlabeled_target',
    pipeline=weak_pipelines,
    pipeline2=train_pipelines2,
    shot=3,
    min_len=60000,
    builder=dict(
        samples_per_gpu=24,
    )
)

target_val_datasets = dict(
    type=single_dataset_type,
    name=target_domain,
    split='validation_target',
    pipeline=test_pipelines,
    shot=3,
    builder=dict(
        samples_per_gpu=128,
    )
)

target_test_datasets = dict(
    type=single_dataset_type,
    name=target_domain,
    split='unlabeled_target',
    pipeline=test_pipelines,
    shot=3,
    builder=dict(
        samples_per_gpu=128,
    )
)

train_datasets = {
    'batch_size': 1,
    'pipeline': train_pipelines,
    0: source_datasets,
    1: target_labeled_datasets,
    2: target_unlabeled_datasets,
}

test_datasets = {
    'batch_size': 128,
    0: target_test_datasets,
    1: target_val_datasets,
}

datasets = dict(
    n_workers=4,
    train=train_datasets,
    test=test_datasets,
)
