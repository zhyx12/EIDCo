backbone_optimizer = dict(
    type='SGD',
    lr=0.004,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='vit_grl_basenet',
    use_bottleneck=True,
    bottleneck_dim=1024,
    width=1024,
    optimizer=backbone_optimizer,
)

classifier_optimizer = dict(
    type='SGD',
    lr=0.004,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov=True,
)

classifier = dict(
    type='vit_classifier',
    width=1024,
    class_num=65,
    optimizer=classifier_optimizer,
)

scheduler = dict(
    type='InvLR',
    gamma=0.001,
    power=0.75,
)

models = dict(
    base_model=backbone,
    classifier=classifier,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
