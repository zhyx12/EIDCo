_base_ = [
    '../_base_/cls_datasets/ssda_officehome_3shot/triple_aug_officehome_2gpu_prob_5e1_CP.py',
    '../_base_/cls_models/ssda_officehome_toalign_srcmix/resnet_34_toalign_srcmix_temp_0.05.py'
]

log_interval = 100
val_interval = 500

control = dict(
    log_interval=log_interval,
    max_iters=20000,
    val_interval=val_interval,
    cudnn_deterministic=False,
    save_interval=500,
    max_save_num=1,
    # seed=2,
)

train = dict(
    toalign=True,
    lambda_info_nce=0.5,
    lambda_fixmatch=1.0,
    lambda_kld=0.1,
    lambda_temp=0.3,
    prob_threshold=0.90,
    src_mixup=True,
    beta_param=1.0,
    src_ce_type='strong',
    loss_type=1,
)

test = dict(
    custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred'),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='target_pred'),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST")
    ]
)
