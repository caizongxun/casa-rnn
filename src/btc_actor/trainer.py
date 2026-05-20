"""
trainer.py -- ActorTrainer  (v2)

Fix log
-------
* Confidence loss is now delayed: only added after epoch >= conf_warmup_epoch
  (default 5).  In early training the direction loss needs to dominate;
  adding confidence loss immediately hurts convergence.
* Gradient norm is now printed every epoch for diagnostics.
* LR warm-up added: linear ramp over first 3 epochs before cosine decay.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .actor import BTCActor


def build_windows(
    ohlcv: np.ndarray,
    window: int = 168,
    horizon: int = 4,
    flat_threshold: float = 0.003,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Xs, ys = [], []
    close = ohlcv[:, 3]
    for i in range(window, len(ohlcv) - horizon):
        x   = ohlcv[i - window: i]
        ret = (close[i + horizon] - close[i]) / (close[i] + 1e-8)
        if   ret >  flat_threshold: label = 0
        elif ret < -flat_threshold: label = 2
        else:                       label = 1
        Xs.append(x)
        ys.append(label)
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(ys,           dtype=torch.long)
    return X, y


def compute_reward(
    action: torch.Tensor,
    log_ret: torch.Tensor,
    confidence: torch.Tensor,
    fee: float = 0.001,
) -> torch.Tensor:
    sign     = torch.where(action == 0,  torch.ones_like(log_ret),
               torch.where(action == 2, -torch.ones_like(log_ret),
                           torch.zeros_like(log_ret)))
    fee_mask = (action != 1).float() * fee
    return sign * log_ret * confidence - fee_mask


class ActorTrainer:

    def __init__(
        self,
        model: BTCActor,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model    = model.to(device)
        self.device   = device
        self.lr       = lr
        self.opt      = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0

    def _set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def pretrain(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
        epochs:  int = 30,
        batch_size: int = 128,
        patience: int = 7,
        warmup_epochs: int = 3,
        conf_warmup_epoch: int = 5,
    ):
        print("\n=== Stage 1: Supervised Pre-training ===")
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
        label_counts = torch.bincount(y_train)
        print(f"  Label dist: LONG={label_counts[0]} FLAT={label_counts[1]} SHORT={label_counts[2]}")
        print(f"  LR warm-up: {warmup_epochs} epochs | conf loss starts: ep{conf_warmup_epoch}")

        weights  = 1.0 / (label_counts.float() + 1)
        weights  = (weights / weights.sum() * 3).to(self.device)
        ce_loss  = nn.CrossEntropyLoss(weight=weights)

        train_dl = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl   = DataLoader(TensorDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)

        # Cosine scheduler (starts after warmup)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(1, epochs - warmup_epochs), eta_min=self.lr * 0.05
        )

        no_improve = 0
        for epoch in range(1, epochs + 1):

            # LR warm-up: linear ramp
            if epoch <= warmup_epochs:
                self._set_lr(self.lr * epoch / warmup_epochs)

            self.model.train()
            total_loss, total_gnorm, n = 0.0, 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, conf, _ = self.model(xb)

                loss = ce_loss(logits, yb)

                # Confidence loss only after warm-up period
                if epoch >= conf_warmup_epoch:
                    correct = (logits.argmax(dim=1) == yb).float().detach()
                    loss = loss + 0.1 * F.binary_cross_entropy(conf.squeeze(1), correct)

                self.opt.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                total_loss  += loss.item() * len(xb)
                total_gnorm += gnorm.item()
                n           += len(xb)

            if epoch > warmup_epochs:
                scheduler.step()

            val_loss, val_acc = self._eval_supervised(val_dl, ce_loss)
            avg_loss  = total_loss  / n
            avg_gnorm = total_gnorm / len(train_dl)
            cur_lr    = self.opt.param_groups[0]["lr"]
            gate_vals = torch.sigmoid(self.model.encoder.gate.gate).detach()

            print(f"[Ep{epoch:02d}] "
                  f"train={avg_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.3f}  "
                  f"gnorm={avg_gnorm:.3f}  lr={cur_lr:.2e}  "
                  f"gate(mean={gate_vals.mean():.3f} std={gate_vals.std():.3f})")

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
    def _eval_supervised(self, dl, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits, _, _ = self.model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            correct    += (logits.argmax(dim=1) == yb).sum().item()
            n          += len(xb)
        return total_loss / n, correct / n

    def ppo_finetune(
        self,
        ohlcv_train: np.ndarray,
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

        close        = ohlcv_train[:, 3]
        log_ret_full = np.zeros(len(ohlcv_train))
        for i in range(len(ohlcv_train) - horizon):
            log_ret_full[i] = math.log(
                (close[i + horizon] + 1e-8) / (close[i] + 1e-8)
            )

        valid_idx = list(range(window, len(ohlcv_train) - horizon))
        ohlcv_t   = torch.tensor(ohlcv_train, dtype=torch.float32)

        # PPO uses its own lower LR
        ppo_opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        best_mean_reward = -float("inf")

        for update in range(1, n_updates + 1):
            self.model.eval()
            idxs = np.random.choice(valid_idx, size=batch_size, replace=False)

            with torch.no_grad():
                Xb    = torch.stack([ohlcv_t[i - window: i] for i in idxs]).to(self.device)
                lr_b  = torch.tensor([log_ret_full[i] for i in idxs],
                                     dtype=torch.float32).to(self.device)
                logits_old, conf_old, values_old = self.model(Xb)
                probs_old     = F.softmax(logits_old, dim=-1)
                dist_old      = torch.distributions.Categorical(probs_old)
                actions       = dist_old.sample()
                log_probs_old = dist_old.log_prob(actions)
                rewards       = compute_reward(actions, lr_b, conf_old.squeeze(1))
                advantages    = (rewards - values_old.squeeze(1)).detach()
                advantages    = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns       = rewards.detach()

            self.model.train()
            for _ in range(ppo_epochs):
                logits_new, conf_new, values_new = self.model(Xb)
                probs_new     = F.softmax(logits_new, dim=-1)
                dist_new      = torch.distributions.Categorical(probs_new)
                log_probs_new = dist_new.log_prob(actions)
                entropy       = dist_new.entropy().mean()
                ratio         = torch.exp(log_probs_new - log_probs_old)
                surr1         = ratio * advantages
                surr2         = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                policy_loss   = -torch.min(surr1, surr2).mean()
                value_loss    = F.mse_loss(values_new.squeeze(1), returns)
                loss          = policy_loss + value_coef * value_loss - entropy_coef * entropy

                ppo_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                ppo_opt.step()

            mean_reward = rewards.mean().item()
            if update % 10 == 0 or update == 1:
                gate_std = torch.sigmoid(self.model.encoder.gate.gate).std().item()
                print(f"[PPO u{update:04d}] "
                      f"reward={mean_reward:+.5f}  "
                      f"conf={conf_old.mean().item():.3f}  "
                      f"gate_std={gate_std:.4f}  "
                      f"p_loss={policy_loss.item():.4f}")

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.model.save(self.ckpt_dir / "best_ppo.pt")

        print(f"  Best mean reward: {best_mean_reward:+.6f}")
        print(f"  Final checkpoint: {self.ckpt_dir}/best_ppo.pt")
