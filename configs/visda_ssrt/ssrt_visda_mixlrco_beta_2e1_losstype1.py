_base_ = [
    '../_base_/cls_datasets/uda_visda/uda_visda_4gpu_for_ssrt.py',
    '../_base_/cls_models/visda_uda/ssrt_vit_b_visda_temp_3e2.py'
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
    lambda_info_nce=0.5,
    lambda_fixmatch=1.0,
    lambda_kld=0.1,
    lambda_temp=0.7,
    prob_threshold=0.98,
    src_mixup=True,
    beta_param=0.2,
    loss_type=1,
)

test = dict(
custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred',class_acc=True),
        dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred',class_acc=True),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST")
    ]
)
