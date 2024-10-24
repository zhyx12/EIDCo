backbone_optimizer = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='sfda_srcmix_contrastive_model',
    model_dict=dict(
        type='SFDAResNetBase',
        resnet_name='ResNet50',
        bottleneck_dim=256,
    ),
    classifier_dict=dict(
        type='SFDAClassifier',
        num_class=65,
        bottleneck_dim=256,
        temp=0.03,
    ),
    num_class=65,
    low_dim=256,
    model_moving_average_decay=0.99,
    fusion_type='reconstruct_double_detach',
    extra_bank_size=512,
    high_img_size=1,
    optimizer=backbone_optimizer,
)


scheduler = dict(
    type='InvLR',
    gamma=0.001,
    power=0.75,
)

models = dict(
    base_model=backbone,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
