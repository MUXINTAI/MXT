#!/bin/bash
cd "E:\PycharmProjects\NTIRE2025_CDFSOD"
# 激活conda环境
echo "激活cdfsod环境..."
# 获取conda路径并激活环境
CONDA_BASE=$(conda info --base 2>/dev/null | sed 's/\\/\//g')
if [ ! -z "$CONDA_BASE" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
  conda activate cdfsod 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "警告: 无法激活cdfsod环境，使用系统默认Python"
  else
    echo "成功激活cdfsod环境"
  fi
else
  echo "警告: 无法获取conda路径，使用系统默认Python"
fi

source E:/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate cdfsod

datalist=(
#dataset1
dataset2
# dataset3
)
shot_list=(
1
#5
# 10
)
model_list=(
"l"
#"b"
#"s"
)
for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do
      echo "训练 ${dataset} ${shot}shot 基础模型 (vit${model})..."
      CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS weights/trained/vit${model}_0089999.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/vit${model}_base/${dataset}_${shot}shot/
    done
  done
done
