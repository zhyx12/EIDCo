backbone_optimizer = dict(
    type='SGD',
    lr=0.0003,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='hda_srcmix_contrastive_model',
    model_dict=dict(
        type='GVBResNetBase',
        resnet_name='ResNet50',
    ),
    classifier_dict=dict(
        type='HDAClassifier',
        num_class=12,
        inc=2048,
        temp=0.03,
    ),
    num_class=12,
    low_dim=2048,
    model_moving_average_decay=0.99,
    fusion_type='reconstruct_double_detach',
    extra_bank_size=512,
    high_img_size=1,
    optimizer=backbone_optimizer,
)

discriminator_optimizer = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

discriminator = dict(
    type='GVBAdversarialNetwork',
    in_feature=12,
    hidden_size=1024,
    optimizer=discriminator_optimizer,
)

scheduler = dict(
    type='InvLR',
    gamma=0.001,
    power=0.75,
)

models = dict(
    base_model=backbone,
    discriminator=discriminator,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
