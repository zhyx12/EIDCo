backbone_optimizer = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='srcmix_contrastive_model',
    model_dict=dict(
        type='resnet_with_fc1',
        depth=34,
    ),
    classifier_dict=dict(
        type='Classifier_deep_without_fc1',
        num_class=65,
        inc=512,
        temp=0.05,
    ),
    num_class=65,
    low_dim=512,
    fusion_type='reconstruct_double_detach',
    extra_bank_size=512,
    high_img_size=1,
    optimizer=backbone_optimizer,
    # find_unused_parameters=True,
)

scheduler = dict(
    type='InvLR',
    gamma=0.0001,
    power=0.75,
)

models = dict(
    base_model=backbone,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
    # sync_bn=True,
    # broadcast_buffers=False,
)
