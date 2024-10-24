backbone_optimizer = dict(
    type='SGD',
    lr=0.002,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov=True,
)

backbone = dict(
    type='ssrt_srcmix_contrastive_model',
    model_dict=dict(
        type='vit_grl_basenet',
        use_bottleneck=True,
        bottleneck_dim=1024,
        width=1024,
    ),
    classifier_dict=dict(
        type='vit_cosine_classifier',
        width=1024,
        class_num=12,
        temp=0.03,
    ),
    num_class=12,
    low_dim=1024,
    model_moving_average_decay=0.99,
    fusion_type='reconstruct_double_detach',
    extra_bank_size=512,
    high_img_size=1,
    select_src_by_tgt_similarity=True,
    src_keep_ratio=0.5,
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
