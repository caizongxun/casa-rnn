"""
trainer.py -- ActorTrainer  (v5)

Fixes
-----
* PPO inner-loop (ppo_epochs > 1) caused ratio collapse on batch_size=64.
  Replaced with REINFORCE + moving-average baseline (no clipping needed,
  numerically stable, works on small batches).
* Reward log now shows mean AND std so you can see if the distribution is
  collapsing to zero variance.
* FLAT-collapse detection kept (fires if >90% FLAT for 20 consecutive updates).
* Action distribution (L/F/S) printed every 10 updates.
* entropy_coef=0.05 to keep exploration alive on CPU runs.
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
        patience: int = 10,
        warmup_epochs: int = 3,
        conf_warmup_epoch: int = 8,
        label_smoothing: float = 0.1,
    ):
        print("\n=== Stage 1: Supervised Pre-training ===")
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
        lc = torch.bincount(y_train)
        print(f"  Label dist: LONG={lc[0]} FLAT={lc[1]} SHORT={lc[2]}")
        print(f"  {'Ep':>4}  {'tr_loss':>8}  {'vl_loss':>8}  {'acc':>5}  "
              f"{'LONG':>5} {'FLAT':>5} {'SHORT':>5}  {'gnorm':>6}  lr")

        weights = 1.0 / (lc.float() + 1)
        weights = (weights / weights.sum() * 3).to(self.device)
        ce_w    = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        ce_uw   = nn.CrossEntropyLoss()

        train_dl = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(1, epochs - warmup_epochs), eta_min=self.lr * 0.05
        )

        no_improve = 0
        for epoch in range(1, epochs + 1):
            if epoch <= warmup_epochs:
                self._set_lr(self.lr * epoch / warmup_epochs)

            self.model.train()
            total_ce, total_gnorm, n = 0.0, 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, conf, _ = self.model(xb)
                ce   = ce_w(logits, yb)
                loss = ce
                if epoch >= conf_warmup_epoch:
                    correct = (logits.argmax(1) == yb).float().detach()
                    loss = loss + 0.1 * F.binary_cross_entropy(conf.squeeze(1), correct)
                self.opt.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_ce    += ce.item() * len(xb)
                total_gnorm += gnorm.item()
                n           += len(xb)

            if epoch > warmup_epochs:
                scheduler.step()

            vl_loss, val_acc, pc = self._eval(val_dl, ce_uw)
            tr_loss = total_ce / n
            avg_gn  = total_gnorm / len(train_dl)
            cur_lr  = self.opt.param_groups[0]["lr"]

            print(f"  {epoch:>4d}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
                  f"{val_acc:>5.3f}  "
                  f"{pc[0]:>5.3f} {pc[1]:>5.3f} {pc[2]:>5.3f}  "
                  f"{avg_gn:>6.3f}  {cur_lr:.2e}")

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
    def _eval(self, dl, criterion) -> Tuple[float, float, list]:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []
        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits, _, _ = self.model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(yb.cpu())
            n += len(xb)
        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc    = (preds == labels).float().mean().item()
        pc = []
        for c in range(3):
            m = labels == c
            pc.append((preds[m] == c).float().mean().item() if m.sum() > 0 else 0.0)
        return total_loss / n, acc, pc

    def ppo_finetune(
        self,
        ohlcv_train: np.ndarray,
        window: int = 168,
        horizon: int = 4,
        n_updates: int = 500,
        batch_size: int = 256,
        entropy_coef: float = 0.05,
        value_coef: float = 0.3,
        flat_threshold: float = 0.003,
        flat_collapse_patience: int = 20,
        baseline_momentum: float = 0.95,
    ):
        """
        REINFORCE + moving-average baseline (replaces unstable PPO inner loop).

        One gradient step per update. No clipping, no old/new ratio.
        Stable on small CPU batches.
        """
        print("\n=== Stage 2: REINFORCE Fine-tuning ===")
        print(f"  window={window}  horizon={horizon}  n_updates={n_updates}")
        print(f"  batch={batch_size}  entropy_coef={entropy_coef}  value_coef={value_coef}")
        print(f"  {'Update':>7}  {'r_mean':>8}  {'r_std':>7}  "
              f"{'L':>4} {'F':>4} {'S':>4}  {'conf':>6}  {'entr':>6}  {'gnorm':>6}")

        close        = ohlcv_train[:, 3]
        log_ret_full = np.zeros(len(ohlcv_train))
        for i in range(len(ohlcv_train) - horizon):
            log_ret_full[i] = math.log(
                (close[i + horizon] + 1e-8) / (close[i] + 1e-8)
            )

        valid_idx = list(range(window, len(ohlcv_train) - horizon))
        ohlcv_t   = torch.tensor(ohlcv_train, dtype=torch.float32)

        rf_opt = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        best_mean_reward = -float("inf")
        flat_streak      = 0
        baseline         = 0.0   # moving average baseline

        for update in range(1, n_updates + 1):
            idxs = np.random.choice(valid_idx, size=batch_size, replace=False)

            self.model.train()
            Xb   = torch.stack([ohlcv_t[i - window: i] for i in idxs]).to(self.device)
            lr_b = torch.tensor([log_ret_full[i] for i in idxs],
                                 dtype=torch.float32).to(self.device)

            logits, conf, values = self.model(Xb)
            probs   = F.softmax(logits, dim=-1)
            dist    = torch.distributions.Categorical(probs)
            actions = dist.sample()
            entropy = dist.entropy().mean()

            with torch.no_grad():
                rewards = compute_reward(actions, lr_b, conf.squeeze(1))

            # moving-average baseline
            batch_mean = rewards.mean().item()
            baseline   = baseline_momentum * baseline + (1 - baseline_momentum) * batch_mean
            advantages = (rewards - baseline).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs   = dist.log_prob(actions)
            policy_loss = -(log_probs * advantages).mean()
            value_loss  = F.mse_loss(values.squeeze(1), rewards.detach())
            loss        = policy_loss + value_coef * value_loss - entropy_coef * entropy

            rf_opt.zero_grad()
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            rf_opt.step()

            r_mean  = rewards.mean().item()
            r_std   = rewards.std().item()
            ad      = torch.bincount(actions, minlength=3)

            # FLAT-collapse detection
            flat_frac = ad[1].item() / batch_size
            flat_streak = flat_streak + 1 if flat_frac > 0.90 else 0
            if flat_streak >= flat_collapse_patience:
                print(f"  [RL] FLAT collapse at update {update} -- stopping.")
                break

            if update % 10 == 0 or update == 1:
                print(f"  {update:>7d}  {r_mean:>+8.5f}  {r_std:>7.5f}  "
                      f"{ad[0]:>4d} {ad[1]:>4d} {ad[2]:>4d}  "
                      f"{conf.mean().item():>6.3f}  "
                      f"{entropy.item():>6.3f}  "
                      f"{gnorm:>6.3f}")

            if r_mean > best_mean_reward:
                best_mean_reward = r_mean
                self.model.save(self.ckpt_dir / "best_ppo.pt")

        print(f"  Best mean reward: {best_mean_reward:+.6f}")
        print(f"  Checkpoint: {self.ckpt_dir}/best_ppo.pt")
