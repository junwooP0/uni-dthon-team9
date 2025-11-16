#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Baseline Model
"""

class CFG:
    # Core Hyperparameters
    IMG_SIZE: int = 512
    EPOCHS: int = 1
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 64
    SEED: int = 42
    DIM: int = 256
    NUM_WORKERS: int = 32
    NO_PRETRAIN: bool = False  # True → disable ImageNet weights
    BACKBONE: str = "efficientnet_v2_s"   # ["resnet18", "resnet34", "resnet50", "efficientnet_b3", "efficientnet_v2_s", "clip_vit_b16"]
    
    LOSS_TYPE: str = "giou" # 가능값: "smooth_l1", "giou"

    # Paths
    TRAIN_JSON_DIR: str = "/home/jwmoon/workspace/jwmoon/unid/train_valid"  # train + valid 폴더 포함
    TRAIN_JPG_DIR: str = "/home/jwmoon/workspace/jwmoon/unid/train_valid"
    VAL_JSON_DIR: str = None  # validation 없이 학습
    VAL_JPG_DIR: str = None
    TEST_JSON_DIR: str = "/raid/slpr/jwmoon/unid/test/query"  # 매 epoch 제출용
    TEST_JPG_DIR: str = "/raid/slpr/jwmoon/unid/test/images"

    # Output Paths
    CKPT_PATH: str = "./checkpoints/baseline_cross_attn.pth"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"
