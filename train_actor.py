"""
train_actor.py  v4 -- full upgrade pipeline

Pipeline
--------
  Stage 0 : Masked-patch self-supervised pre-training
  Stage 1 : Supervised classification
             - Focal loss + Mixup + Curriculum
             - Regime auxiliary loss (HMM)
             - Dynamic flat threshold (ATR-based)
             - Optional walk-forward CV
  Stage 2 : PPO fine-tuning
             - Rollout buffer 2048
             - Sharpe-aware reward + stop-loss penalty
             - Backtest-in-the-loop every 50 updates
  Ensemble: 3 seeds trained sequentially (OOM-safe)

Usage
-----
    python train_actor.py                      # full pipeline, 3-seed ensemble
    python train_actor.py --seeds 1            # single model (faster)
    python train_actor.py --no-wfcv            # skip walk-forward CV
    python train_actor.py --pretrain-only
    python train_actor.py --ppo-only
    python train_actor.py --d-model 256 --n-layers 4 --n-heads 8  # GPU
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from casa_rnn.market_data import load_and_enrich
from casa_rnn.funding import load_or_fetch_futures_features
fro