# 域适应增强模块 (Domain Adaptation Enhancement Module, DAEM)

## 概述

域适应增强模块 (DAEM) 是针对跨域少样本目标检测任务设计的创新组件，专注于改善模型在目标域上的泛化能力。该模块通过多种机制处理域差异，使模型能够更有效地从源域迁移知识到目标域，即使在只有少量目标域样本的情况下。

## 核心功能

DAEM 模块提供以下核心功能：

1. **批次增强**：通过风格迁移和数据增强技术，使训练样本更加多样化，减少过拟合风险。
   - 风格适应：使源域图像的风格更接近目标域
   - 颜色抖动：随机调整图像的亮度、对比度和饱和度

2. **特征对齐**：通过最大平均差异 (MMD) 损失，减少源域和目标域特征分布之间的差异，促进域不变特征的学习。

3. **原型精炼**：动态更新和改进类原型表示，使其能够更好地捕获目标域的特征分布。
   - 使用动量更新策略，平滑原型变化
   - 从检测框中提取高质量特征进行原型更新

4. **注意力适应**：通过自适应注意力机制，突出域不变特征，抑制域特定特征。

## 安装和依赖

DAEM 模块已集成到 NTIRE2025_CDFSOD 项目中，依赖关系已包含在项目的 requirements.txt 文件中。

## 使用方法

### 1. 使用单独的脚本运行 DAEM 增强训练

```bash
bash run_daem.sh
```

这个脚本会使用我们专门设计的域适应增强模块进行训练，而不影响原始的训练脚本。

### 2. 配置文件

我们为每个数据集和 shot 设置都提供了专用的 DAEM 配置文件，例如：

- `configs/dataset1/vitl_shot1_dataset1_finetune_daem.yaml`
- `configs/dataset2/vitl_shot1_dataset2_finetune_daem.yaml`
- `configs/dataset3/vitl_shot1_dataset3_finetune_daem.yaml`

这些配置文件继承自原始配置，并添加了 DAEM 特定的参数设置。

### 3. 手动调用训练脚本

您也可以直接调用增强版训练脚本，并指定配置文件：

```bash
python tools/train_enhanced_net.py \
  --num-gpus 1 \
  --config-file configs/dataset1/vitl_shot1_dataset1_finetune_daem.yaml \
  MODEL.WEIGHTS weights/trained/vitl_0089999.pth \
  DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
  OUTPUT_DIR output/daem_vitl/dataset1_1shot/
```

## 参数配置

DAEM 模块提供了多个参数，可以通过配置文件进行调整：

- `DAEM.ENABLED`：是否启用 DAEM 模块
- `DAEM.STRENGTH`：域适应增强的强度 (0.0-1.0)
- `DAEM.FEATURE_ALIGNMENT`：是否启用特征对齐
- `DAEM.PROTOTYPE_REFINE`：是否启用原型精炼
- `DAEM.ATTENTION_ADAPT`：是否启用自适应注意力
- `DAEM.MMD_WEIGHT`：MMD 损失的权重
- `DAEM.STYLE_WEIGHT`：风格一致性损失的权重
- `DAEM.PROTOTYPE_MOMENTUM`：原型更新的动量
- `DAEM.ATTENTION_COEF`：注意力适应系数

## 文件结构

```
|- lib/
|  |- daem/
|  |  |- __init__.py         # 模块初始化
|  |  |- config.py           # 配置定义
|  |  |- daem_module.py      # 核心模块实现
|
|- tools/
|  |- train_enhanced_net.py  # 增强版训练脚本
|
|- configs/
|  |- dataset1/
|  |  |- vitl_shot1_dataset1_finetune_daem.yaml  # DAEM 配置
|  |- dataset2/
|  |  |- vitl_shot1_dataset2_finetune_daem.yaml  # DAEM 配置
|  |- dataset3/
|  |  |- vitl_shot1_dataset3_finetune_daem.yaml  # DAEM 配置
|
|- run_daem.sh               # DAEM 运行脚本
```

## 性能提升

与基线模型相比，DAEM 模块可以显著提高在跨域少样本目标检测任务上的性能：

- 更好的域适应能力
- 更有效的知识迁移
- 更高的检测精度
- 更好的泛化性能

## 技术细节

DAEM 实现了几个技术创新点：

1. **渐进式域适应**：随着训练的进行，自动调整域适应的强度
2. **多尺度风格调整**：通过调整图像风格的多个方面，使样本更接近目标域
3. **特征层面的对齐**：在特征空间中对齐源域和目标域，而不仅仅是像素级别
4. **自适应注意力机制**：学习对域不变特征给予更高的注意力权重

## 贡献与问题反馈

如果您有任何问题或改进建议，请联系项目维护者。 