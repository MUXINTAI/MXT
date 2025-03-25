#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Domain Adaptation Enhancement Module (DAEM) configuration.
"""

from detectron2.config import CfgNode as CN

def add_daem_config(cfg):
    """
    Add config for Domain Adaptation Enhancement Module (DAEM).
    """
    # Unfreeze config to add new fields
    cfg.defrost()
    
    # Add DAEM configuration
    cfg.DAEM = CN()
    
    # Basic parameters
    cfg.DAEM.ENABLED = True              # Enable DAEM
    cfg.DAEM.STRENGTH = 0.8              # Strength of enhancement application
    
    # Feature alignment
    cfg.DAEM.FEATURE_ALIGNMENT = True    # Enable feature alignment
    cfg.DAEM.MMD_WEIGHT = 0.16           # MMD loss weight
    cfg.DAEM.STYLE_WEIGHT = 0.12         # Style transfer weight
    
    # Prototype refinement
    cfg.DAEM.PROTOTYPE_REFINE = True     # Enable prototype refinement
    cfg.DAEM.PROTOTYPE_MOMENTUM = 0.9    # Momentum for prototype updates
    
    # Attention adaptation
    cfg.DAEM.ATTENTION_ADAPT = True      # Enable attention adaptation
    cfg.DAEM.ATTENTION_COEF = 0.5        # Attention adaptation coefficient
    
    # Learning parameters
    cfg.DAEM.WARM_UP_ITERS = 500         # Warm-up iterations
    cfg.DAEM.LR_MULTIPLIER = 1.0         # DAEM learning rate multiplier
    cfg.DAEM.BACKBONE_LR_FACTOR = 1.0    # Backbone learning rate factor
    cfg.DAEM.BIAS_LR_FACTOR = 2.0        # Bias learning rate factor
    
    # Freezing strategy
    cfg.DAEM.FREEZE_BACKBONE = False     # Don't freeze backbone network
    cfg.DAEM.FREEZE_RPN = False          # Don't freeze RPN network
    cfg.DAEM.FREEZE_ROI_HEADS = False    # Don't freeze ROI heads
    
    # Contrastive learning
    cfg.DAEM.CONTRASTIVE_ENABLED = False  # Enable contrastive learning
    cfg.DAEM.CONTRASTIVE_TEMP = 0.1       # Temperature parameter for contrastive learning
    cfg.DAEM.CONT_INSTANCE_WEIGHT = 0.5   # Instance-level contrastive learning weight
    cfg.DAEM.CONT_DOMAIN_WEIGHT = 0.3     # Domain-level contrastive learning weight
    cfg.DAEM.CONT_PROTOTYPE_WEIGHT = 0.2  # Prototype-level contrastive learning weight
    cfg.DAEM.ADAPTIVE_TEMP = False        # Adaptive temperature
    cfg.DAEM.HARD_NEGATIVE_MINING = False # Hard negative mining
    cfg.DAEM.CROSS_DOMAIN_CONTRAST = False # Cross-domain contrastive learning
    cfg.DAEM.DOMAIN_PROMPTER_RATIO = 1.0   # Domain prompter quantity multiplier
    cfg.DAEM.PROMPTER_TEMP = 1.0          # Domain prompter contrastive learning temperature
    cfg.DAEM.PROTOTYPE_TEMP = 1.0         # Prototype contrastive learning temperature
    
    # Domain prompter
    cfg.DAEM.DOMAIN_PROMPTER_ENABLED = False  # Enable domain prompter
    
    # Advanced features
    cfg.DAEM.FEATURE_DECOUPLING = False   # Feature decoupling and recombination
    cfg.DAEM.MULTI_LEVEL_ADAPT = False    # Multi-level adaptation
    cfg.DAEM.HARD_MINING = False          # Hard sample mining
    cfg.DAEM.HARD_MINING_RATIO = 0.3      # Hard sample ratio
    cfg.DAEM.CONTEXT_ENHANCE = False      # Context enhancement
    cfg.DAEM.UNCERTAINTY_GUIDED = False   # Uncertainty-guided approach
    cfg.DAEM.UNCERTAINTY_THRESHOLD = 0.7  # Uncertainty threshold
    cfg.DAEM.ADVERSARIAL_TRAINING = False # Adversarial training
    cfg.DAEM.ADV_WEIGHT = 0.1             # Adversarial loss weight
    
    # Refreeze config
    cfg.freeze()
    
    return cfg 