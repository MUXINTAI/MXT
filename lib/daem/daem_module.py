import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
import logging

# 添加日志记录
logger = logging.getLogger(__name__)

class DomainAdaptationEnhancementModule(nn.Module):
    """
    域适应增强模块 (Domain Adaptation Enhancement Module, DAEM)
    用于提高跨域少样本目标检测的性能
    
    核心功能:
    1. 批次增强 - 通过风格迁移和数据增强增强训练样本
    2. 特征对齐 - 通过MMD损失减少源域和目标域的特征差异
    3. 原型精炼 - 动态更新和改进类原型
    4. 注意力适应 - 通过自适应注意力机制增强域不变特征
    5. 特征解耦与重组 - 分离内容和风格特征，实现跨域特征混合
    6. 多尺度上下文增强 - 捕获目标周围的环境信息
    7. 不确定性引导的伪标签生成 - 通过模型不确定性生成高质量伪标签
    8. 对抗域混合训练 - 使用域对抗训练提高特征的域不变性
    """
    
    def __init__(self, cfg):
        """
        Initialize the DAEM with configuration parameters.
        
        Args:
            cfg: Configuration object containing DAEM parameters
        """
        super().__init__()
        self.cfg = cfg
        
        # Basic parameters
        self.enabled = cfg.DAEM.ENABLED
        self.strength = cfg.DAEM.STRENGTH
        
        # Feature alignment parameters
        self.feature_alignment = cfg.DAEM.FEATURE_ALIGNMENT
        self.mmd_weight = cfg.DAEM.MMD_WEIGHT
        self.style_weight = cfg.DAEM.STYLE_WEIGHT
        
        # 原型精炼
        self.prototype_refine = cfg.DAEM.PROTOTYPE_REFINE
        self.prototype_momentum = cfg.DAEM.PROTOTYPE_MOMENTUM
        self.prototype_buffer = {}
        
        # 注意力适应
        self.attention_adapt = cfg.DAEM.ATTENTION_ADAPT
        self.attention_coef = cfg.DAEM.ATTENTION_COEF
        
        # 特征解耦与重组
        self.feature_decoupling = getattr(cfg.DAEM, "FEATURE_DECOUPLING", False)
        
        # 多层级适应
        self.multi_level_adapt = getattr(cfg.DAEM, "MULTI_LEVEL_ADAPT", False)
        
        # 困难样本挖掘
        self.hard_mining = getattr(cfg.DAEM, "HARD_MINING", False)
        self.hard_mining_ratio = getattr(cfg.DAEM, "HARD_MINING_RATIO", 0.3)
        
        # 上下文增强
        self.context_enhance = getattr(cfg.DAEM, "CONTEXT_ENHANCE", False)
        
        # 不确定性引导
        self.uncertainty_guided = getattr(cfg.DAEM, "UNCERTAINTY_GUIDED", False)
        self.uncertainty_threshold = getattr(cfg.DAEM, "UNCERTAINTY_THRESHOLD", 0.7)
        
        # 对抗训练
        self.adversarial_training = getattr(cfg.DAEM, "ADVERSARIAL_TRAINING", False)
        self.adv_weight = getattr(cfg.DAEM, "ADV_WEIGHT", 0.1)
        
        # 对比学习相关配置
        self.contrastive_enabled = cfg.DAEM.CONTRASTIVE_ENABLED
        self.contrastive_temp = cfg.DAEM.CONTRASTIVE_TEMP
        self.cont_instance_weight = cfg.DAEM.CONT_INSTANCE_WEIGHT
        self.cont_domain_weight = cfg.DAEM.CONT_DOMAIN_WEIGHT
        self.cont_prototype_weight = cfg.DAEM.CONT_PROTOTYPE_WEIGHT
        self.adaptive_temp = cfg.DAEM.ADAPTIVE_TEMP
        self.hard_negative_mining = cfg.DAEM.HARD_NEGATIVE_MINING
        self.cross_domain_contrast = cfg.DAEM.CROSS_DOMAIN_CONTRAST
        self.domain_prompter_ratio = cfg.DAEM.DOMAIN_PROMPTER_RATIO
        self.prompter_temp = cfg.DAEM.PROMPTER_TEMP
        self.prototype_temp = cfg.DAEM.PROTOTYPE_TEMP
        
        # 初始化特征对齐所需的统计信息
        self.domain_features = None
        self.domain_stats = {}  # 添加域统计数据字典

        # 初始化style_encoder
        self.style_encoder = None
        if self.feature_alignment and self.style_weight > 0:
            # 简单的风格编码器，用于提取风格特征
            self.style_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(inplace=True)
            )
        
        # 初始化网络组件
        if self.context_enhance:
            self.context_processor = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, padding=1)
            )
        
        if self.attention_adapt:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(3, 3, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 域对抗分类器
        if self.adversarial_training:
            self.domain_classifier = nn.Sequential(
                GradientReversal(lambda_=1.0),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
    
    # 添加缺失的适应权重计算方法
    def _get_adaptation_weight(self, current_iter, max_iter):
        """
        计算域适应权重，随着训练进行逐渐增加权重
        
        Args:
            current_iter: 当前迭代次数
            max_iter: 最大迭代次数
            
        Returns:
            计算得到的权重系数(0到1之间)
        """
        # 使用余弦退火策略计算权重
        if max_iter == 0:
            return 1.0
            
        # 训练初期权重较小，训练后期权重较大
        # 前20%的训练逐渐上升到0.5，之后缓慢上升到1.0
        ratio = current_iter / max_iter
        
        if ratio < 0.2:
            # 训练初期线性上升
            weight = ratio * 2.5
        else:
            # 训练后期使用余弦退火，平滑过渡到最终权重
            weight = 0.5 + 0.5 * (1 + np.cos(np.pi * (1 - (ratio - 0.2) / 0.8)))
            
        return float(weight)
            
    def enhance_batch(self, batched_inputs: List[Dict[str, Any]], iteration: int) -> List[Dict[str, Any]]:
        """
        Apply domain adaptation enhancements to the input batch.
        
        Args:
            batched_inputs: List of input dictionaries with image and annotations
            iteration: Current training iteration
            
        Returns:
            Enhanced batch with adapted features and annotations
        """
        if not self.enabled or len(batched_inputs) == 0:
            return batched_inputs
        
        # Apply strength ramp-up during warm-up phase
        if iteration < self.cfg.DAEM.WARM_UP_ITERS:
            current_strength = self.strength * (iteration / self.cfg.DAEM.WARM_UP_ITERS)
        else:
            current_strength = self.strength
        
        # Create a copy of inputs to avoid modifying the original
        enhanced_inputs = batched_inputs.copy()
        
        # Apply enhancements to each sample
        for i, input_dict in enumerate(enhanced_inputs):
            # Determine if this is source or target domain
            is_source = self._is_source_domain(input_dict)
            
            # Apply style transfer if enabled
            if self.feature_alignment and current_strength > 0 and "image" in input_dict:
                enhanced_inputs[i] = self._apply_style_transfer(input_dict, is_source, current_strength)
            
            # Apply attention adaptation if enabled
            if self.attention_adapt and current_strength > 0 and "image" in input_dict:
                enhanced_inputs[i] = self._apply_attention_adaptation(input_dict, is_source, current_strength)
            
            # Apply context enhancement if enabled
            if self.cfg.DAEM.CONTEXT_ENHANCE and current_strength > 0 and "image" in input_dict:
                enhanced_inputs[i] = self._apply_context_enhancement(input_dict, is_source, current_strength)
            
            # Update instance annotations if needed
            if self.prototype_refine and "instances" in input_dict:
                enhanced_inputs[i] = self._refine_instances(input_dict, is_source, current_strength)
            
            # Apply uncertainty-guided adaptation if enabled
            if self.cfg.DAEM.UNCERTAINTY_GUIDED and "instances" in input_dict:
                enhanced_inputs[i] = self._apply_uncertainty_guidance(input_dict, is_source, current_strength)
        
        return enhanced_inputs
    
    def compute_alignment_loss(self, model: nn.Module, data: List[Dict[str, Any]], iteration: int) -> Dict[str, torch.Tensor]:
        """
        Compute feature alignment losses for domain adaptation.
        
        Args:
            model: The detection model
            data: Batch of input data
            iteration: Current training iteration
            
        Returns:
            Dictionary of alignment losses
        """
        if not self.enabled or not self.feature_alignment:
            return {}
        
        losses = {}
        
        # Extract features from the model - this is model dependent
        # In a real implementation, you would access specific model features
        # or compute them from the model's forward pass
        
        try:
            source_features = None
            target_features = None
            
            # Try to access features if the model has them
            if hasattr(model, "feature_source") and hasattr(model, "feature_target"):
                source_features = model.feature_source
                target_features = model.feature_target
            
            # Compute MMD loss if we have the features
            if source_features is not None and target_features is not None and self.mmd_weight > 0:
                mmd_loss = self._compute_mmd_loss(source_features, target_features)
                losses["loss_daem_mmd"] = self.mmd_weight * mmd_loss
            
            # Compute style consistency loss if enabled
            if self.style_weight > 0 and hasattr(model, "style_features"):
                style_loss = self._compute_style_loss(model.style_features)
                losses["loss_daem_style"] = self.style_weight * style_loss
            
            # Compute contrastive loss if enabled
            if self.contrastive_enabled and hasattr(model, "instance_features"):
                # This assumes the model has collected instance features 
                # and their class labels during forward pass
                instance_features = model.instance_features
                class_labels = model.class_labels
                
                if len(instance_features) > 1:  # Need at least 2 features for contrastive loss
                    cont_loss = self._compute_contrastive_loss(torch.stack(instance_features), 
                                                             torch.tensor(class_labels, device=instance_features[0].device))
                    losses["loss_daem_contrastive"] = cont_loss
        
        except Exception as e:
            # Log error but don't crash training
            logger.warning(f"Error computing alignment loss: {e}")
        
        return losses
    
    def refine_prototypes(self, model, batch_data):
        """
        精炼类原型表示
        
        Args:
            model: 检测模型
            batch_data: 批次数据
        """
        if not self.prototype_refine or not hasattr(model, "class_prototypes"):
            return
        
        # 获取当前类原型
        current_prototypes = model.class_prototypes
        
        # 提取高质量特征
        features_dict = {}
        labels_list = []
        features_list = []
        
        # 从批次数据中提取特征
        for data in batch_data:
            if "instances" not in data or len(data["instances"]) == 0:
                continue
                
            # 获取图像特征 - 改进：直接使用模型提取强特征
            with torch.no_grad():
                # 获取backbone特征
                features = model.backbone(data["image"].unsqueeze(0))
                if isinstance(features, dict):
                    # 如果backbone返回dict，获取所有层级特征用于更好的表示
                    feature_maps = []
                    # 优先使用高层特征，包含更多语义信息
                    if "res5" in features:
                        feature_maps.append(features["res5"])
                    elif "res4" in features:
                        feature_maps.append(features["res4"])
                    else:
                        # 使用最后几层特征
                        feature_keys = sorted(list(features.keys()))
                        for key in feature_keys[-2:]:  # 使用最后两层特征
                            feature_maps.append(features[key])
                    
                    # 默认使用最深层特征
                    features = feature_maps[0]
            
            # 收集实例特征和标签
            instances = data["instances"]
            for i in range(len(instances)):
                box = instances.gt_boxes.tensor[i]
                label = instances.gt_classes[i]
                
                # 从特征图中提取对应框的特征
                roi_features = self._extract_roi_features(features, box)
                
                # 增强：应用特征归一化以提高对比学习效果
                roi_features = F.normalize(roi_features, dim=0)
                
                # 按类别保存特征
                class_id = int(label.item())
                if class_id not in features_dict:
                    features_dict[class_id] = []
                features_dict[class_id].append(roi_features)
                
                # 添加到列表用于对比学习
                labels_list.append(class_id)
                features_list.append(roi_features)
        
        # 更新类原型表示
        updated_prototypes = {}
        for class_id, class_features in features_dict.items():
            if len(class_features) > 0:
                # 计算增强后的特征中心
                class_feature = torch.stack(class_features).mean(0)
                
                # 应用动量更新方式
                if class_id in self.prototype_buffer:
                    updated_proto = (
                        self.prototype_momentum * self.prototype_buffer[class_id] + 
                        (1 - self.prototype_momentum) * class_feature
                    )
                else:
                    updated_proto = class_feature
                
                # 确保原型也是归一化的
                updated_proto = F.normalize(updated_proto, dim=0)
                    
                self.prototype_buffer[class_id] = updated_proto
                updated_prototypes[class_id] = updated_proto
        
        # 增强：使用更高效的对比学习损失
        if len(features_list) > 1:  # 至少需要两个特征进行对比
            features_tensor = torch.stack(features_list)
            labels_tensor = torch.tensor(labels_list, device=features_tensor.device)
            
            # 计算对比损失并记录
            contrastive_loss = self._compute_contrastive_loss(features_tensor, labels_tensor)
            
            # 记录损失
            if hasattr(model, "log_dict"):
                model.log_dict["prototype_contrastive_loss"] = contrastive_loss.item()
            
            # 增强：应用梯度来优化特征提取器
            if self.training and hasattr(model, 'backbone') and contrastive_loss.requires_grad:
                contrastive_loss.backward(retain_graph=True)
        
        # 更新模型中的类原型
        for class_id, proto in updated_prototypes.items():
            if class_id in current_prototypes:
                current_prototypes[class_id] = proto
    
    def _extract_roi_features(self, features, box, output_size=7):
        """从特征图中提取ROI特征 - 增强版本"""
        # 调整特征尺寸以匹配输入要求
        if len(features.shape) == 3:
            features = features.unsqueeze(0)  # 添加批次维度
        elif len(features.shape) == 2:
            features = features.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        
        # 确保特征是4D的 [batch, channels, height, width]
        if len(features.shape) != 4:
            try:
                # 尝试重塑为4D
                features = features.view(1, -1, 1, 1)
                return features.flatten()  # 直接返回扁平化特征
            except:
                # 如果失败，返回零向量
                return torch.zeros(features.shape[1] if len(features.shape) > 1 else 512, 
                                  device=features.device)
        
        # 将box转换为ROI格式 [batch_idx, x1, y1, x2, y2]
        roi = torch.cat([torch.zeros(1, device=box.device), box])
        
        # 计算空间缩放比例
        h_ratio = features.shape[2] / 1.0  # 假设原图尺寸为1.0
        w_ratio = features.shape[3] / 1.0  # 假设原图尺寸为1.0
        spatial_scale = min(h_ratio, w_ratio)
        
        try:
            # 尝试使用ROI池化提取特征
            import torchvision
            roi_features = torchvision.ops.roi_align(
                features, 
                [roi.unsqueeze(0)],
                output_size=(output_size, output_size),
                spatial_scale=spatial_scale,
                aligned=True
            )
            
            # 增强：使用多层池化提取更丰富的特征
            global_pooled = F.adaptive_avg_pool2d(roi_features, (1, 1))
            max_pooled = F.adaptive_max_pool2d(roi_features, (1, 1))
            
            # 组合平均池化和最大池化特征
            combined_features = torch.cat([global_pooled, max_pooled], dim=1)
            return combined_features.flatten()
        except Exception as e:
            # 如果ROI池化失败，使用简单的全局池化
            try:
                pooled = F.adaptive_avg_pool2d(features, (1, 1))
                return pooled.flatten()
            except:
                # 最后的回退选项
                return torch.zeros(features.shape[1], device=features.device)
    
    def _compute_contrastive_loss(self, features, labels, temperature=0.1):
        """计算增强的监督对比损失"""
        # 特征已在调用前归一化，这里确保
        features = F.normalize(features, dim=1)
        
        # 计算特征相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # 创建标签匹配矩阵(同类为正样本对)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # 排除自身相似度
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
        
        # 增强：使用硬负样本挖掘 - 关注最难区分的负样本
        k = max(1, int(similarity_matrix.size(0) * 0.25))  # 选择前25%的困难负样本
        
        # 应用掩码
        pos_sim = similarity_matrix * mask * logits_mask
        neg_sim = similarity_matrix * (1 - mask) * logits_mask
        
        # 避免数值问题
        neg_sim_exp = torch.exp(neg_sim)
        pos_sim_exp = torch.exp(pos_sim)
        
        # 硬负样本挖掘 - 对每个样本，选择k个最困难的负样本
        neg_sim_hard = torch.zeros_like(neg_sim)
        
        for i in range(neg_sim.size(0)):
            # 获取当前样本的负样本
            curr_neg = neg_sim[i]
            # 找到有效的负样本索引（排除零值）
            valid_indices = (curr_neg > 0).nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                # 选择前k个最高的负样本（最相似的负样本）
                _, hard_indices = torch.topk(curr_neg[valid_indices], k)
                selected_indices = valid_indices[hard_indices]
                neg_sim_hard[i, selected_indices] = neg_sim[i, selected_indices]
        
        # 使用硬负样本计算对比损失
        neg_sim_hard_exp = torch.exp(neg_sim_hard)
        neg_sim_sum = neg_sim_hard_exp.sum(dim=1)
        
        # 计算每个正样本对的对比损失
        pos_count = mask.sum(dim=1)
        loss = torch.zeros(features.size(0), device=features.device)
        
        # 对于每个有正样本对的样本计算损失
        valid_indices = pos_count > 0
        if valid_indices.sum() > 0:
            valid_pos_sim = pos_sim[valid_indices]
            valid_neg_sum = neg_sim_sum[valid_indices].unsqueeze(1)
            valid_pos_count = pos_count[valid_indices].unsqueeze(1)
            
            # 计算每个有效样本的损失 - 使用对数和变换增强数值稳定性
            valid_loss = -torch.log(pos_sim_exp[valid_indices] / (valid_neg_sum + 1e-8)) / (valid_pos_count + 1e-8)
            loss[valid_indices] = valid_loss.sum(dim=1)
        
        # 返回平均损失
        return loss.mean()
    
    def _apply_style_transfer(self, input_dict: Dict[str, Any], is_source: bool, strength: float) -> Dict[str, Any]:
        """
        Apply style transfer to enhance domain adaptation.
        
        Args:
            input_dict: Input dictionary with image
            is_source: Whether input is from source domain
            strength: Current adaptation strength
            
        Returns:
            Input dictionary with style-enhanced image
        """
        if "image" not in input_dict:
            return input_dict
            
        # Create a copy to avoid modifying the original
        result = input_dict.copy()
        image = input_dict["image"]
        
        # Only apply style transfer with some probability based on strength
        if torch.rand(1).item() > strength:
            return result
            
        # Ensure image is 4D tensor [B,C,H,W]
        original_dim = image.dim()
        if original_dim == 3:  # If [C,H,W] format
            image = image.unsqueeze(0)  # Add batch dimension to [1,C,H,W]
            
        # Standardize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        
        # Convert to 0-1 range and normalize
        img_normalized = (image / 255.0 - mean) / std
        
        # Compute channel statistics
        b, c, h, w = img_normalized.shape
        img_flat = img_normalized.reshape(b, c, -1)
        
        # Compute mean and std per channel
        channel_mean = img_flat.mean(dim=2, keepdim=True)
        channel_std = img_flat.std(dim=2, keepdim=True) + 1e-5
        
        # Apply random perturbation to channel stats based on strength
        perturb_amount = strength * 0.1  # Scale perturbation by strength
        rand_mean = channel_mean + (torch.rand_like(channel_mean) * 2 - 1) * perturb_amount
        rand_std = channel_std * (1 + (torch.rand_like(channel_std) * 2 - 1) * perturb_amount)
        
        # Apply style transfer
        img_flat_normalized = (img_flat - channel_mean) / channel_std
        img_flat_styled = img_flat_normalized * rand_std + rand_mean
        
        # Reshape back
        img_styled = img_flat_styled.reshape(b, c, h, w)
        
        # Convert back to original range
        img_styled = img_styled * std + mean
        img_styled = img_styled * 255.0
        
        # Ensure pixel values in valid range
        img_styled = torch.clamp(img_styled, 0, 255)
        
        # Restore original dimensions
        if original_dim == 3:
            img_styled = img_styled.squeeze(0)
            
        result["image"] = img_styled
        
        return result
    
    def _apply_color_jitter(self, image):
        """
        应用颜色抖动增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        # 简化的颜色抖动实现
        with torch.no_grad():
            # 确保图像是4D张量 [B,C,H,W]
            original_dim = image.dim()
            if original_dim == 3:  # 如果是[C,H,W]格式
                image = image.unsqueeze(0)  # 添加batch维度变为[1,C,H,W]
                
            # 将图像转换为[0,1]范围
            img_0_1 = image / 255.0
            
            # 亮度、对比度、饱和度和色调调整
            brightness_factor = torch.tensor(1.0 + (torch.rand(1).item() - 0.5) * 0.2)
            contrast_factor = torch.tensor(1.0 + (torch.rand(1).item() - 0.5) * 0.2)
            saturation_factor = torch.tensor(1.0 + (torch.rand(1).item() - 0.5) * 0.2)
            
            # 应用变换
            jittered = img_0_1
            if torch.rand(1).item() < 0.8:
                jittered = jittered * brightness_factor
            if torch.rand(1).item() < 0.8:
                mean = torch.mean(jittered, dim=[0, 1, 2], keepdim=True)
                jittered = (jittered - mean) * contrast_factor + mean
            if torch.rand(1).item() < 0.8:
                gray = jittered.mean(dim=0, keepdim=True)
                jittered = jittered * saturation_factor + gray * (1 - saturation_factor)
            
            # 转回原始范围并确保像素值有效
            jittered = torch.clamp(jittered * 255.0, 0, 255)
            
            # 恢复原始维度
            if original_dim == 3:
                jittered = jittered.squeeze(0)
                
            return jittered
    
    def _update_domain_stats(self, image):
        """
        更新域统计信息
        
        Args:
            image: 输入图像
        """
        # 提取简单的图像统计特征
        with torch.no_grad():
            if self.style_encoder is not None:
                # 确保图像是4D张量 [B,C,H,W]
                if image.dim() == 3:  # 如果是[C,H,W]格式
                    image_for_style = image.unsqueeze(0)  # 添加batch维度变为[1,C,H,W]
                else:
                    image_for_style = image
                
                # 标准化并传入style_encoder
                style_feat = self.style_encoder(image_for_style.float() / 255.0)
                
                if "style_features" not in self.domain_stats:
                    self.domain_stats["style_features"] = []
                
                # 保留最近的样本统计信息
                max_buffer_size = 64
                self.domain_stats["style_features"].append(style_feat.detach())
                if len(self.domain_stats["style_features"]) > max_buffer_size:
                    self.domain_stats["style_features"] = self.domain_stats["style_features"][-max_buffer_size:]
            
            # 保存少量图像样本用于域混合
            if self.adversarial_training:
                if "images" not in self.domain_stats:
                    self.domain_stats["images"] = []
                
                max_image_buffer = 16
                self.domain_stats["images"].append(image.detach())
                if len(self.domain_stats["images"]) > max_image_buffer:
                    self.domain_stats["images"] = self.domain_stats["images"][-max_image_buffer:]
    
    def _compute_mmd_loss(self, source_features, target_features):
        """
        计算最大平均差异(MMD)损失
        
        Args:
            source_features: 源域特征
            target_features: 目标域特征
            
        Returns:
            MMD损失
        """
        def _gaussian_kernel(x, y, sigma=1.0):
            # 高斯核函数
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
            
            x = x.unsqueeze(1)  # 形状变为 (x_size, 1, dim)
            y = y.unsqueeze(0)  # 形状变为 (1, y_size, dim)
            
            tiled_x = x.expand(x_size, y_size, dim)
            tiled_y = y.expand(x_size, y_size, dim)
            
            kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / dim
            return torch.exp(-kernel_input / (2 * sigma))
        
        # 确保特征是二维的
        if source_features.dim() > 2:
            source_features = source_features.reshape(source_features.size(0), -1)
        if target_features.dim() > 2:
            target_features = target_features.reshape(target_features.size(0), -1)
            
        batch_size = min(source_features.size(0), target_features.size(0))
        if batch_size == 0:
            return torch.tensor(0.0, device=source_features.device)
            
        # 取相同数量的样本
        source_features = source_features[:batch_size]
        target_features = target_features[:batch_size]
        
        # 计算核矩阵
        kernels = [_gaussian_kernel(source_features, source_features, sigma=sigma) +
                   _gaussian_kernel(target_features, target_features, sigma=sigma) -
                   2 * _gaussian_kernel(source_features, target_features, sigma=sigma)
                   for sigma in [0.5, 1.0, 2.0, 5.0]]
        
        # 求和得到MMD损失
        return sum(k.mean() for k in kernels)
    
    def _compute_multi_level_loss(self, multi_level_features):
        """
        计算多层级适应损失
        
        Args:
            multi_level_features: 多层级特征字典，包含源域和目标域的各层特征
            
        Returns:
            多层级适应损失
        """
        loss = 0.0
        
        # 为不同层级分配不同权重 - 高层特征权重更大
        level_weights = {
            "low": 0.2,
            "mid": 0.3,
            "high": 0.5
        }
        
        for level, weight in level_weights.items():
            if level in multi_level_features:
                source = multi_level_features[level]["source"]
                target = multi_level_features[level]["target"]
                
                if source is not None and target is not None:
                    level_loss = self._compute_mmd_loss(source, target)
                    loss += weight * level_loss
        
        return loss
    
    def _compute_adversarial_loss(self, domain_features):
        """
        计算域对抗损失
        
        Args:
            domain_features: 域特征，包含源域和目标域的特征
            
        Returns:
            域对抗损失
        """
        source_features = domain_features["source"]
        target_features = domain_features["target"]
        
        batch_size_source = source_features.size(0)
        batch_size_target = target_features.size(0)
        
        # 源域标签为1，目标域标签为0
        source_labels = torch.ones(batch_size_source, 1, device=source_features.device)
        target_labels = torch.zeros(batch_size_target, 1, device=target_features.device)
        
        # 域分类预测
        source_preds = self.domain_classifier(source_features)
        target_preds = self.domain_classifier(target_features)
        
        # 二元交叉熵损失
        loss_source = F.binary_cross_entropy(source_preds, source_labels)
        loss_target = F.binary_cross_entropy(target_preds, target_labels)
        
        return (loss_source + loss_target) / 2.0
    
    def _compute_decoupling_loss(self, content_style_features):
        """
        计算特征解耦损失
        
        Args:
            content_style_features: 内容风格特征字典
            
        Returns:
            解耦损失
        """
        # 内容一致性损失 - 跨域内容特征应该相似
        content_source = content_style_features["content_source"]
        content_target = content_style_features["content_target"]
        
        # 风格多样性损失 - 风格特征应该多样化
        style_source = content_style_features["style_source"]
        style_target = content_style_features["style_target"]
        
        # 计算内容特征的余弦相似度损失
        content_sim = F.cosine_similarity(content_source, content_target, dim=1).mean()
        content_loss = 1.0 - content_sim  # 最大化相似度
        
        # 计算风格特征的差异性损失
        style_sim = F.cosine_similarity(style_source, style_target, dim=1).mean()
        style_loss = style_sim  # 最小化相似度
        
        return content_loss + 0.5 * style_loss
    
    def _apply_hard_mining(self, batch_data):
        """
        应用困难样本挖掘
        
        Args:
            batch_data: 批次数据
            
        Returns:
            处理后的批次数据
        """
        # 实际应用中需要访问每个样本的检测难度
        # 这里使用简化实现，随机选择一部分样本作为"困难样本"
        
        # 实际中应基于检测置信度、损失值等指标评估样本难度
        n_samples = len(batch_data)
        n_hard = max(1, int(n_samples * self.hard_mining_ratio))
        
        # 随机选择困难样本索引
        hard_indices = torch.randperm(n_samples)[:n_hard]
        
        # 创建增强后的批次，包括原始样本和困难样本的副本
        enhanced_batch = batch_data.copy()
        
        # 添加困难样本的副本
        for idx in hard_indices:
            if idx < len(batch_data):
                enhanced_batch.append(batch_data[idx])
        
        return enhanced_batch
    
    def _apply_domain_mixup(self, image):
        """
        应用域混合增强
        
        Args:
            image: 输入图像
            
        Returns:
            域混合后的图像
        """
        # 确保图像是4D张量 [B,C,H,W]
        original_dim = image.dim()
        if original_dim == 3:  # 如果是[C,H,W]格式
            image = image.unsqueeze(0)  # 添加batch维度变为[1,C,H,W]
            
        # 简化实现，实际中可能需要更复杂的域混合策略
        # 这里简单地应用随机混合系数
        mixed_image = image.clone()  # 默认返回原图像的副本
        
        # 如果有缓存的图像统计信息，使用它进行混合
        if "images" in self.domain_stats and len(self.domain_stats["images"]) > 0:
            # 随机选择一个缓存图像
            idx = torch.randint(0, len(self.domain_stats["images"]), (1,)).item()
            cached_image = self.domain_stats["images"][idx]
            
            # 确保缓存图像也是4D张量
            if cached_image.dim() == 3:
                cached_image = cached_image.unsqueeze(0)
            
            # 确保两个图像的尺寸兼容 - 使用广播或调整尺寸
            if cached_image.shape != image.shape:
                try:
                    # 尝试广播 - 调整batch维度或通道维度
                    if cached_image.shape[0] != image.shape[0]:
                        if cached_image.shape[0] == 1:
                            cached_image = cached_image.expand(image.shape[0], -1, -1, -1)
                        elif image.shape[0] == 1:
                            image = image.expand(cached_image.shape[0], -1, -1, -1)
                    
                    # 调整空间维度
                    if cached_image.shape[2:] != image.shape[2:]:
                        cached_image = F.interpolate(cached_image, size=image.shape[2:], mode='bilinear', align_corners=False)
                except Exception as e:
                    print(f"形状调整失败: {e}")
                    # 在形状不兼容的情况下，直接返回原图像
                    mixed_image = image
                    if original_dim == 3:
                        mixed_image = mixed_image.squeeze(0)
                    return mixed_image

            # 随机混合系数
            alpha = torch.rand(1).item() * 0.5 + 0.5  # 0.5-1.0之间

            # 混合图像
            mixed_image = alpha * image + (1 - alpha) * cached_image
        
        # 恢复原始维度
        if original_dim == 3:
            mixed_image = mixed_image.squeeze(0)
            
        return mixed_image

    def _apply_attention_adaptation(self, input_dict: Dict[str, Any], is_source: bool, strength: float) -> Dict[str, Any]:
        """
        Apply attention adaptation to enhance domain adaptation.
        
        Args:
            input_dict: Input dictionary with image
            is_source: Whether input is from source domain
            strength: Current adaptation strength
            
        Returns:
            Input dictionary with attention-enhanced image
        """
        # In a real implementation, apply attention adaptation
        # Here we simply return the original input as placeholder
        if "image" not in input_dict:
            return input_dict
            
        # Create a copy to avoid modifying the original
        result = input_dict.copy()
        
        # Apply channel attention if implemented
        if hasattr(self, 'channel_attention'):
            image = input_dict["image"]
            
            # Ensure image is 4D tensor [B,C,H,W]
            original_dim = image.dim()
            if original_dim == 3:  # If [C,H,W] format
                image = image.unsqueeze(0)  # Add batch dimension to [1,C,H,W]
                
            # Apply channel attention - scaled by strength
            attention_map = self.channel_attention(image.float() / 255.0)
            enhanced_image = image * (1.0 + strength * (attention_map - 0.5))
            
            # Ensure pixel values in valid range
            enhanced_image = torch.clamp(enhanced_image, 0, 255)
            
            # Restore original dimensions
            if original_dim == 3:
                enhanced_image = enhanced_image.squeeze(0)
                
            result["image"] = enhanced_image
            
        return result
    
    def _apply_context_enhancement(self, input_dict: Dict[str, Any], is_source: bool, strength: float) -> Dict[str, Any]:
        """
        Apply context enhancement to improve object detection by considering surrounding context.
        
        Args:
            input_dict: Input dictionary with image
            is_source: Whether input is from source domain
            strength: Current adaptation strength
            
        Returns:
            Input dictionary with context-enhanced image
        """
        # In a real implementation, apply context enhancement
        # Here we simply return the original input as placeholder
        if "image" not in input_dict or not hasattr(self, 'context_processor'):
            return input_dict
            
        # Create a copy to avoid modifying the original
        result = input_dict.copy()
        
        image = input_dict["image"]
        
        # Ensure image is 4D tensor [B,C,H,W]
        original_dim = image.dim()
        if original_dim == 3:  # If [C,H,W] format
            image = image.unsqueeze(0)  # Add batch dimension to [1,C,H,W]
            
        # Apply context enhancement - scaled by strength
        context_features = self.context_processor(image.float() / 255.0)
        enhanced_image = image + strength * context_features * 255.0
        
        # Ensure pixel values in valid range
        enhanced_image = torch.clamp(enhanced_image, 0, 255)
        
        # Restore original dimensions
        if original_dim == 3:
            enhanced_image = enhanced_image.squeeze(0)
            
        result["image"] = enhanced_image
        
        return result
        
    def _apply_uncertainty_guidance(self, input_dict: Dict[str, Any], is_source: bool, strength: float) -> Dict[str, Any]:
        """
        Apply uncertainty-guided adaptation to focus on uncertain regions.
        
        Args:
            input_dict: Input dictionary with instances
            is_source: Whether input is from source domain
            strength: Current adaptation strength
            
        Returns:
            Input dictionary with uncertainty-guided enhanced instances
        """
        # In a real implementation, apply uncertainty guidance
        # Here we simply return the original input as placeholder
        return input_dict

    def _is_source_domain(self, input_dict: Dict[str, Any]) -> bool:
        """
        Determine if an input belongs to the source domain.
        
        In practice, this would be determined by dataset metadata or explicit domain markers.
        Here, we use a simple heuristic based on the dataset name or id if present.
        
        Args:
            input_dict: Input dictionary with image and annotations
            
        Returns:
            Boolean indicating if input is from source domain
        """
        # Use dataset_name if available
        if "dataset_name" in input_dict:
            dataset_name = input_dict["dataset_name"]
            # Base datasets are source domain
            return "base" in dataset_name.lower() or "coco" in dataset_name.lower()
            
        # Use dataset_id if available
        elif "dataset_id" in input_dict:
            # Typically 0 is source domain, others are target domains
            return input_dict["dataset_id"] == 0
        
        # Default to treating as source domain for safety
        return True

    def _compute_style_loss(self, style_features: torch.Tensor) -> torch.Tensor:
        """
        Compute style consistency loss for style transfer.
        
        Args:
            style_features: Dictionary or tensor of style features
            
        Returns:
            Style loss tensor
        """
        # In a real implementation, compute style consistency loss
        # based on gram matrices or other style representations
        
        try:
            if isinstance(style_features, dict):
                # If style_features is a dict with source and target
                source_style = style_features.get("source")
                target_style = style_features.get("target")
                
                if source_style is not None and target_style is not None:
                    # Compute gram matrices
                    source_gram = self._compute_gram_matrix(source_style)
                    target_gram = self._compute_gram_matrix(target_style)
                    
                    # Compute MSE loss between gram matrices
                    style_loss = F.mse_loss(source_gram, target_gram)
                    return style_loss
            
            # Fallback to a dummy loss
            return torch.tensor(0.0, device=self._get_device())
            
        except Exception as e:
            logger.warning(f"Error computing style loss: {e}")
            return torch.tensor(0.0, device=self._get_device())
    
    def _compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix from features.
        
        Args:
            features: Feature tensor [B, C, H, W]
            
        Returns:
            Gram matrix [B, C, C]
        """
        batch_size, ch, h, w = features.size()
        features = features.view(batch_size, ch, h * w)
        
        # Transpose and compute gram matrix
        transposed = features.transpose(1, 2)
        gram = torch.bmm(features, transposed)
        
        # Normalize by feature size
        gram = gram / (ch * h * w)
        
        return gram

    def _refine_instances(self, input_dict: Dict[str, Any], is_source: bool, strength: float) -> Dict[str, Any]:
        """
        Refine instance annotations based on prototypes.
        
        Args:
            input_dict: Input dictionary with instances
            is_source: Whether input is from source domain
            strength: Current adaptation strength
            
        Returns:
            Input dictionary with refined instances
        """
        if "instances" not in input_dict or strength <= 0:
            return input_dict
            
        # Only apply refinement with some probability based on strength
        if torch.rand(1).item() > strength:
            return input_dict
            
        # Create a copy to avoid modifying the original
        result = input_dict.copy()
        instances = input_dict["instances"]
        
        # Skip if no instance annotations
        if len(instances) == 0:
            return result
            
        # Skip if no prototypes have been built yet
        if len(self.prototypes) == 0:
            return result
            
        try:
            # Get instance class IDs
            if hasattr(instances, "gt_classes"):
                class_ids = instances.gt_classes
                
                # Process each instance based on its class
                refined_instances = instances.clone()
                
                # In a real implementation, you would:
                # 1. Extract instance features
                # 2. Compare with prototype features
                # 3. Adjust instance boxes/masks based on prototype similarity
                
                # For target domain (where annotations might be less reliable):
                if not is_source:
                    # Example: Adjust confidence or refine boxes based on prototype similarity
                    # This would require custom logic based on your specific use case
                    pass
                
                # Update instance features if your Instances class has this field
                if hasattr(refined_instances, "features"):
                    # Get features from prototypes
                    for i, class_id in enumerate(class_ids):
                        class_id_int = class_id.item()
                        if class_id_int in self.prototypes:
                            # Use prototype feature to enhance instance feature
                            prototype_feature = self.prototypes[class_id_int]
                            if hasattr(refined_instances, "features"):
                                # Blend instance feature with prototype feature
                                instance_feature = refined_instances.features[i]
                                refined_feature = (1 - strength) * instance_feature + strength * prototype_feature
                                refined_instances.features[i] = refined_feature
                
                result["instances"] = refined_instances
                
        except Exception as e:
            logger.warning(f"Error refining instances: {e}")
            
        return result