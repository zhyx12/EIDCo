#!/usr/bin/env bash
job_id=$1
config_file=$2

project_home='CVPR2022'
cd $HOME'/PycharmProjects/'${project_home} || exit

trainer_class=fixmatchsrcmix
validator_class=fixmatchsrcmix
scripts_path=$HOME'/PycharmProjects/'${project_home}'/experiments/scripts/get_visible_card_num.py'
port_scripts_path=$HOME'/PycharmProjects/'${project_home}'/experiments/scripts/generate_random_port.py'
GPUS=$(python ${scripts_path})
PORT=$(python ${port_scripts_path})

python_file=./train.py
# CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  ${python_file} --task_type cls --job_id ${job_id} --config ${config_file} \
  --trainer ${trainer_class} --validator ${validator_class}
