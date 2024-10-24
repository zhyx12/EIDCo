_base_ = [
    '../_base_/cls_datasets/ssrt_ssda_offiehome_3shot/triple_aug_officehome_CR_4gpu.py',
    '../_base_/cls_models/ssda_officehome_toalign_srcmix/ssrt_vit_b_officehome_temp_0.03.py'
]

log_interval = 100
val_interval = 500

control = dict(
    log_interval=log_interval,
    max_iters=10000,
    val_interval=val_interval,
    cudnn_deterministic=False,
    save_interval=500,
    max_save_num=1,
    # seed=2,
)

train = dict(
    lambda_info_nce=0.0,
    lambda_fixmatch=0.0,
    lambda_kld=0.0,
    lambda_temp=0.3,
    prob_threshold=0.90,
    src_mixup=False,
    beta_param=0.2,
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
