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
        resnet_name='ResNet101',
    ),
    classifier_dict=dict(
        type='HDAClassifier',
        num_class=345,
        inc=2048,
        temp=0.07,
    ),
    num_class=345,
    low_dim=2048,
    model_moving_average_decay=0.99,
    fusion_type='reconstruct_double_detach',
    extra_bank_size=512,
    high_img_size=1,
    optimizer=backbone_optimizer,
)

discriminator_optimizer = dict(
    type='SGD',
    lr=0.003,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

discriminator = dict(
    type='GVBAdversarialNetwork',
    in_feature=345,
    hidden_size=1024,
    optimizer=discriminator_optimizer,
)

scheduler = dict(
    type='InvLR',
    gamma=0.0002,
    power=0.75,
)

models = dict(
    base_model=backbone,
    discriminator=discriminator,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
