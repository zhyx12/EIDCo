backbone_optimizer = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.001,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='hda_srcmix_contrastive_model',
    model_dict=dict(
        type='GVBResNetBase',
        # depth=50,
        resnet_name='ResNet50',
    ),
    classifier_dict=dict(
        type='HDAClassifier',
        num_class=65,
        inc=2048,
        temp=0.05,
    ),
    num_class=65,
    low_dim=2048,
    model_moving_average_decay=0.99,
    fusion_type='reconstruct_double_detach',
    extra_bank_size=512,
    high_img_size=1,
    select_src_by_tgt_similarity=True,
    src_keep_ratio=0.6,
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
    in_feature=65,
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
