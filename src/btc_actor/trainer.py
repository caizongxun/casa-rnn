"""
trainer.py -- ActorTrainer

Two-stage training pipeline:

Stage 1 -- Supervised pre-training
    Objective : predict whether close[t+horizon] > close[t] (LONG),
                within ±threshold (FLAT), or lower (SHORT).
    Loss       : cross-entropy on direction + direction-weighted confidence
    Duration   : until val accuracy plateaus

Stage 2 -- PPO fine-tuning
    Objective  : maximise simulated PnL in a rolling walk-forward window.
    Reward     : log_return * action_sign * confidence  (simple, no lookahead)
    Penalty    : -0.001 per trade (transaction cost)
    Algorithm  : PPO-clip (epsilon=0.2), shared encoder, separate heads
    Duration   : configurable n_ppo_updates

The model is free to discover any internal representation it wants.
We only specify the reward signal -- the encoder evolves on its own.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .actor import BTCActor


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_windows(
    ohlcv: np.ndarray,          # (N, 5)  open high low close volume
    window: int = 168,           # lookback in candles  (168h = 1 week)
    horizon: int = 4,            # predict N candles ahead
    flat_threshold: float = 0.003,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    X : (W, window, 5)  raw OHLCV windows
    y : (W,)            0=LONG  1=FLAT  2=SHORT
    """
    Xs, ys = [], []
    close = ohlcv[:, 3]          # close price column
    for i in range(window, len(ohlcv) - horizon):
        x = ohlcv[i - window: i]                # (window, 5)
        ret = (close[i + horizon] - close[i]) / (close[i] + 1e-8)
        if ret > flat_threshold:
            label = 0   # LONG
        elif ret < -flat_threshold:
            label = 2   # SHORT
        else:
            label = 1   # FLAT
        Xs.append(x)
        ys.append(label)
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    action: torch.Tensor,      # (B,)  0=LONG 1=FLAT 2=SHORT
    log_ret: torch.Tensor,     # (B,)  actual log return over horizon
    confidence: torch.Tensor,  # (B,)  model confidence
    fee: float = 0.001,
) -> torch.Tensor:
    """
    Reward = position_sign * actual_return * confidence - fee_if_trading
    FLAT has zero reward except the transaction-cost penalty doesn't apply.
    """
    sign = torch.where(action == 0,  torch.ones_like(log_ret),
           torch.where(action == 2, -torch.ones_like(log_ret),
                       torch.zeros_like(log_ret)))
    fee_mask = (action != 1).float() * fee
    reward = sign * log_ret * confidence - fee_mask
    return reward


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ActorTrainer:

    def __init__(
        self,
        model: BTCActor,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=1000, eta_min=lr * 0.1
        )
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Stage 1: Supervised pre-training
    # ------------------------------------------------------------------

    def pretrain(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
        epochs:  int = 30,
        batch_size: int = 128,
        patience: int = 5,
    ):
        print("\n=== Stage 1: Supervised Pre-training ===")
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
        label_counts = torch.bincount(y_train)
        print(f"  Label dist: LONG={label_counts[0]} FLAT={label_counts[1]} SHORT={label_counts[2]}")

        # Class weights to handle imbalance
        weights = 1.0 / (label_counts.float() + 1)
        weights = (weights / weights.sum() * 3).to(self.device)
        ce_loss = nn.CrossEntropyLoss(weight=weights)

        train_ds = TensorDataset(X_train, y_train)
        val_ds   = TensorDataset(X_val,   y_val)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        no_improve = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, n = 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, conf, _ = self.model(xb)

                # Direction loss
                loss_dir = ce_loss(logits, yb)

                # Confidence calibration: penalise low confidence on correct predictions
                correct = (logits.argmax(dim=1) == yb).float().detach()
                loss_conf = F.binary_cross_entropy(conf.squeeze(1), correct)

                loss = loss_dir + 0.2 * loss_conf

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                total_loss += loss.item() * len(xb)
                n += len(xb)

            self.scheduler.step()

            # Validation
            val_loss, val_acc = self._eval_supervised(val_dl, ce_loss)
            avg_train = total_loss / n

            # Gate stats -- shows how the model's feature importance evolves
            gate_vals = torch.sigmoid(self.model.encoder.gate.gate).detach()
            gate_info = f"gate mean={gate_vals.mean():.3f} std={gate_vals.std():.3f}"

            print(f"[Pretrain Ep{epoch:02d}] "
                  f"train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"| {gate_info}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save(self.ckpt_dir / "best_pretrain.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        print(f"  Best val acc: {self.best_val_acc:.4f}")

    @torch.no_grad()
    def _eval_supervised(self, dl: DataLoader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits, _, _ = self.model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            correct    += (logits.argmax(dim=1) == yb).sum().item()
            n          += len(xb)
        return total_loss / n, correct / n

    # ------------------------------------------------------------------
    # Stage 2: PPO fine-tuning
    # ------------------------------------------------------------------

    def ppo_finetune(
        self,
        ohlcv_train: np.ndarray,   # (N, 5) raw prices for reward computation
        window: int = 168,
        horizon: int = 4,
        n_updates: int = 200,
        batch_size: int = 64,
        ppo_epochs: int = 4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        flat_threshold: float = 0.003,
    ):
        print("\n=== Stage 2: PPO Fine-tuning ===")
        print(f"  window={window}  horizon={horizon}  n_updates={n_updates}")

        close = ohlcv_train[:, 3]
        log_ret_full = np.zeros(len(ohlcv_train))
        for i in range(len(ohlcv_train) - horizon):
            log_ret_full[i] = math.log(
                (close[i + horizon] + 1e-8) / (close[i] + 1e-8)
            )

        # Collect all valid window indices
        valid_idx = list(range(window, len(ohlcv_train) - horizon))
        ohlcv_t   = torch.tensor(ohlcv_train, dtype=torch.float32)

        best_mean_reward = -float("inf")

        for update in range(1, n_updates + 1):
            # --- Rollout phase: collect trajectories with current policy ---
            self.model.eval()
            idxs = np.random.choice(valid_idx, size=batch_size, replace=False)

            with torch.no_grad():
                Xb = torch.stack([ohlcv_t[i - window: i] for i in idxs]).to(self.device)
                lr_b = torch.tensor(
                    [log_ret_full[i] for i in idxs], dtype=torch.float32
                ).to(self.device)

                logits_old, conf_old, values_old = self.model(Xb)
                probs_old  = F.softmax(logits_old, dim=-1)          # (B, 3)
                dist_old   = torch.distributions.Categorical(probs_old)
                actions    = dist_old.sample()                       # (B,)
                log_probs_old = dist_old.log_prob(actions)           # (B,)

                rewards = compute_reward(
                    actions, lr_b, conf_old.squeeze(1)
                )                                                    # (B,)

                # Advantage (simple: reward - value)
                advantages = (rewards - values_old.squeeze(1)).detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns    = rewards.detach()

            # --- PPO update phase ---
            self.model.train()
            for _ in range(ppo_epochs):
                logits_new, conf_new, values_new = self.model(Xb)
                probs_new  = F.softmax(logits_new, dim=-1)
                dist_new   = torch.distributions.Categorical(probs_new)
                log_probs_new = dist_new.log_prob(actions)
                entropy    = dist_new.entropy().mean()

                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = F.mse_loss(values_new.squeeze(1), returns)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()

            mean_reward = rewards.mean().item()
            mean_conf   = conf_old.mean().item()
            gate_std    = torch.sigmoid(self.model.encoder.gate.gate).std().item()

            if update % 10 == 0 or update == 1:
                print(f"[PPO u{update:04d}] "
                      f"reward={mean_reward:+.5f}  "
                      f"conf={mean_conf:.3f}  "
                      f"gate_std={gate_std:.4f}  "
                      f"policy_loss={policy_loss.item():.4f}")

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.model.save(self.ckpt_dir / "best_ppo.pt")

        print(f"  Best mean reward: {best_mean_reward:+.6f}")
        print(f"  Final checkpoint: {self.ckpt_dir}/best_ppo.pt")
