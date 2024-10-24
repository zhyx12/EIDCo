_base_ = [
    '../_base_/cls_datasets/msda_domainnet/msda_domainnet_target_infograph.py',
    '../_base_/cls_models/msda_domainnet_toalign/msda_resnet_101_toalign_srcmix_contrastive_domainnet_gamma_2e3.py'
]

log_interval = 100
val_interval = 1000

control = dict(
    log_interval=log_interval,
    max_iters=50000,
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
    prob_threshold=0.98,
    src_mixup=True,
    beta_param=1.0,
)

test = dict(
custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST")
    ]
)
