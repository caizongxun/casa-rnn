"""
trainer.py  v6 -- ActorTrainer (full upgrade)

Stages
------
  0: Self-supervised masked-patch pre-training
  1: Supervised classification with:
     - Focal Loss  (replaces CE + class weight)
     - Mixup augmentation
     - Curriculum learning (hard samples phased in)
     - Regime auxiliary loss (HMM labels)
     - Dynamic flat_threshold from ATR
     - Walk-forward cross-validation
  2: PPO fine-tuning with:
     - Proper rollout buffer (2048 steps)
     - Sharpe-aware reward
     - Stop-loss penalty
     - Backtest-in-the-loop every 50 updates
     - Ensemble of 3 seeds trained in sequence (OOM-safe)
"""

from __future__ import annotations

import gc
import math
import copy
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .actor   import BTCActor
from .regime  import fit_regime_labels
from .backtest import run_backtest


# ---------------------------------------------------------------------------
# Dynamic flat threshold
# ---------------------------------------------------------------------------
def dynamic_flat_threshold(
    atr: np.ndarray,
    close: np.ndarray,
    multiplier: float = 0.3,
    lo: float = 0.001,
    hi: float = 0.01,
) -> np.ndarray:
    """Per-bar threshold = clamp(ATR/close * multiplier, lo, hi)"""
    return np.clip(atr / (close + 1e-8) * multiplier, lo, hi)


# ---------------------------------------------------------------------------
# Window builder  (dynamic threshold)
# ---------------------------------------------------------------------------
def build_windows(
    ohlcv: np.ndarray,
    window: int = 168,
    horizon: int = 4,
    flat_threshold: float = 0.003,   # ignored when atr14 col present
    feat_all: Optional[np.ndarray] = None,  # (N, 31) full feature array
    atr_col_idx: int = 9,                   # index of atr14 in feat_all
    regime_labels: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns X (windows of feat_all or ohlcv), y (LONG/FLAT/SHORT),
    and optionally regime_y.
    """
    src    = feat_all if feat_all is not None else ohlcv
    close  = ohlcv[:, 3]
    n      = len(src)

    # per-bar threshold
    if feat_all is not None and feat_all.shape[1] > atr_col_idx:
        atr = feat_all[:, atr_col_idx]
        thresholds = dynamic_flat_threshold(atr, close)
    else:
        thresholds = np.full(n, flat_threshold)

    Xs, ys, rs = [], [], []
    for i in range(window, n - horizon):
        ret = (close[i + horizon] - close[i]) / (close[i] + 1e-8)
        th  = thresholds[i]
        if   ret >  th: label = 0
        elif ret < -th: label = 2
        else:           label = 1
        Xs.append(src[i - window: i])
        ys.append(label)
        if regime_labels is not None:
            rs.append(regime_labels[i])

    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(ys,           dtype=torch.long)
    r = torch.tensor(rs,           dtype=torch.long) if rs else None
    return X, y, r


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------
def mixup_batch(
    xb: torch.Tensor,
    yb: torch.Tensor,
    alpha: float = 0.2,
):
    lam  = np.random.beta(alpha, alpha)
    idx  = torch.randperm(len(xb), device=xb.device)
    x_m  = lam * xb + (1 - lam) * xb[idx]
    # return mixed x + soft label pairs
    return x_m, yb, yb[idx], lam


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------
def compute_reward(
    action:     torch.Tensor,
    log_ret:    torch.Tensor,
    confidence: torch.Tensor,
    roll_std:   torch.Tensor,
    fee:        float = 0.001,
    sl_thresh:  float = 0.015,
    sl_penalty: float = 0.002,
) -> torch.Tensor:
    sign     = torch.where(action == 0,  torch.ones_like(log_ret),
               torch.where(action == 2, -torch.ones_like(log_ret),
                           torch.zeros_like(log_ret)))
    fee_mask = (action != 1).float() * fee
    # Sharpe-aware: scale by rolling volatility
    sharpe_r = sign * log_ret * confidence / (roll_std + 1e-6) - fee_mask
    # Stop-loss penalty: directional trade where loss exceeds sl_thresh
    loss_hit = (sign * log_ret < -sl_thresh).float()
    return sharpe_r - loss_hit * sl_penalty


# ---------------------------------------------------------------------------
# ActorTrainer
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
        self.model    = model.to(device)
        self.device   = device
        self.lr       = lr
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    # -----------------------------------------------------------------------
    # Stage 0: self-supervised masked-patch pre-training
    # -----------------------------------------------------------------------
    def masked_pretrain(
        self,
        X_train: torch.Tensor,
        epochs: int = 5,
        batch_size: int = 128,
        mask_ratio: float = 0.15,
        lr: float = 1e-3,
    ):
        print("\n=== Stage 0: Masked-Patch Self-Supervised Pre-training ===")
        self._set_lr(lr)
        dl = DataLoader(TensorDataset(X_train), batch_size=batch_size,
                        shuffle=True, drop_last=True)
        for epoch in range(1, epochs + 1):
            self.model.train()
            total, n = 0.0, 0
            for (xb,) in dl:
                xb = xb.to(self.device)
                recon, target, mask = self.model.masked_reconstruct(xb, mask_ratio)
                loss = F.mse_loss(recon[mask], target[mask])
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total += loss.item() * len(xb); n += len(xb)
            print(f"  [Stage0 ep{epoch:02d}] recon_loss={total/n:.5f}")
        print("  Stage 0 done.")

    # -----------------------------------------------------------------------
    # Stage 1: supervised classification
    # -----------------------------------------------------------------------
    def pretrain(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
        r_train: Optional[torch.Tensor] = None,
        r_val:   Optional[torch.Tensor] = None,
        epochs:  int   = 30,
        batch_size: int = 128,
        patience: int  = 8,
        warmup_epochs: int = 1,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.2,
        curriculum_start_epoch: int = 3,
        regime_loss_coef: float = 0.05,
    ):
        print("\n=== Stage 1: Supervised Pre-training ===")
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
        lc = torch.bincount(y_train)
        print(f"  Label dist: LONG={lc[0]} FLAT={lc[1]} SHORT={lc[2]}")
        header = (f"  {'Ep':>4}  {'tr_loss':>8}  {'vl_loss':>8}  "
                  f"{'acc':>5}  {'LONG':>5} {'FLAT':>5} {'SHORT':>5}  "
                  f"{'gnorm':>6}  lr")
        print(header)

        weights = 1.0 / (lc.float() + 1)
        weights = (weights / weights.sum() * 3).to(self.device)
        focal   = FocalLoss(gamma=2.0, weight=weights)
        ce_uw   = nn.CrossEntropyLoss()
        regime_ce = nn.CrossEntropyLoss() if r_train is not None else None

        # curriculum: start with easy (non-flat) samples, add flat later
        easy_mask  = (y_train != 1)
        easy_X, easy_y = X_train[easy_mask], y_train[easy_mask]
        easy_r     = r_train[easy_mask] if r_train is not None else None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(1, epochs - warmup_epochs), eta_min=self.lr * 0.05
        )
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # curriculum: phase in flat samples after curriculum_start_epoch
            if epoch < curriculum_start_epoch:
                Xtr, ytr = easy_X, easy_y
                rtr      = easy_r
            else:
                Xtr, ytr = X_train, y_train
                rtr      = r_train

            if epoch == 1:
                self._set_lr(self.lr * 1.0 / max(warmup_epochs, 1))
            elif epoch <= warmup_epochs:
                self._set_lr(self.lr * epoch / warmup_epochs)

            ds  = TensorDataset(*(x for x in [Xtr, ytr, rtr] if x is not None))
            dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_items = [X_val, y_val] + ([r_val] if r_val is not None else [])
            vdl = DataLoader(TensorDataset(*val_items),
                             batch_size=batch_size, shuffle=False)

            self.model.train()
            total_loss, total_gnorm, n = 0.0, 0.0, 0

            for batch in dl:
                if r_train is not None:
                    xb, yb, rb = batch
                else:
                    xb, yb     = batch
                    rb         = None
                xb, yb = xb.to(self.device), yb.to(self.device)

                # Mixup
                xb, ya, yb_mix, lam = mixup_batch(xb, yb, mixup_alpha)
                ya = ya.to(self.device); yb_mix = yb_mix.to(self.device)

                logits, conf, _, regime_logits = self.model(
                    xb, return_regime=True
                )
                # mixed focal loss
                loss = lam * focal(logits, ya) + (1 - lam) * focal(logits, yb_mix)

                # confidence calibration (from ep 5)
                if epoch >= 5:
                    correct = (logits.argmax(1) == ya).float().detach()
                    loss = loss + 0.1 * F.binary_cross_entropy(
                        conf.squeeze(1), correct
                    )

                # regime auxiliary loss
                if regime_ce is not None and rb is not None:
                    rb = rb.to(self.device)
                    loss = loss + regime_loss_coef * regime_ce(regime_logits, rb)

                self.opt.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_loss  += loss.item() * len(xb)
                total_gnorm += gnorm.item()
                n += len(xb)

            if epoch > warmup_epochs:
                scheduler.step()

            vl_loss, val_acc, pc = self._eval(vdl, ce_uw)
            cur_lr  = self.opt.param_groups[0]["lr"]
            print(f"  {epoch:>4d}  {total_loss/n:>8.4f}  {vl_loss:>8.4f}  "
                  f"{val_acc:>5.3f}  "
                  f"{pc[0]:>5.3f} {pc[1]:>5.3f} {pc[2]:>5.3f}  "
                  f"{total_gnorm/len(dl):>6.3f}  {cur_lr:.2e}")

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
    def _eval(self, dl, criterion):
        self.model.eval()
        total_loss, n = 0.0, 0
        all_p, all_l  = [], []
        for batch in dl:
            xb, yb = batch[0].to(self.device), batch[1].to(self.device)
            logits, _, _ = self.model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            all_p.append(logits.argmax(1).cpu())
            all_l.append(yb.cpu())
            n += len(xb)
        preds  = torch.cat(all_p)
        labels = torch.cat(all_l)
        acc    = (preds == labels).float().mean().item()
        pc     = [
            (preds[labels == c] == c).float().mean().item()
            if (labels == c).any() else 0.0 for c in range(3)
        ]
        return total_loss / n, acc, pc

    # -----------------------------------------------------------------------
    # Walk-forward cross-validation
    # -----------------------------------------------------------------------
    def walk_forward_cv(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        r: Optional[torch.Tensor],
        n_folds: int = 0,           # 0 = auto (target ~1 month test per fold)
        min_train_frac: float = 0.5,
        epochs: int = 20,
        batch_size: int = 128,
    ) -> List[float]:
        """
        Rolling walk-forward CV.
        If n_folds=0, auto-selects fold size so each test window ~ 720 bars
        (30 days of 1h data) or 1/12 of total data, whichever is smaller.
        """
        N = len(X)
        if n_folds == 0:
            test_size = min(720, N // 12)
            n_folds   = max(2, (N - int(N * min_train_frac)) // test_size)
        else:
            test_size = (N - int(N * min_train_frac)) // n_folds

        print(f"\n=== Walk-forward CV: {n_folds} folds, test_size={test_size} ===")
        accs = []
        for fold in range(n_folds):
            train_end = int(N * min_train_frac) + fold * test_size
            test_end  = min(train_end + test_size, N)
            if test_end <= train_end:
                break

            Xt, yt = X[:train_end], y[:train_end]
            Xv, yv = X[train_end:test_end], y[train_end:test_end]
            rt     = r[:train_end]  if r is not None else None
            rv     = r[train_end:test_end] if r is not None else None

            # fresh model copy for each fold
            fold_model = copy.deepcopy(self.model)
            fold_trainer = ActorTrainer(
                fold_model, self.device, self.lr,
                checkpoint_dir=str(self.ckpt_dir / f"fold_{fold}")
            )
            fold_trainer.pretrain(
                Xt, yt, Xv, yv, rt, rv,
                epochs=epochs, batch_size=batch_size,
                patience=5, warmup_epochs=1, label_smoothing=0.05,
            )
            accs.append(fold_trainer.best_val_acc)
            print(f"  Fold {fold+1}/{n_folds}  val_acc={fold_trainer.best_val_acc:.4f}")
            del fold_model, fold_trainer
            gc.collect()
            if self.device != "cpu":
                torch.cuda.empty_cache()

        print(f"  WF-CV mean acc: {np.mean(accs):.4f}  std: {np.std(accs):.4f}")
        return accs

    # -----------------------------------------------------------------------
    # Stage 2: PPO fine-tuning
    # -----------------------------------------------------------------------
    def ppo_finetune(
        self,
        ohlcv_train:  np.ndarray,
        feat_train:   Optional[np.ndarray] = None,
        window:       int   = 168,
        horizon:      int   = 4,
        n_updates:    int   = 500,
        rollout_size: int   = 2048,
        mini_batch:   int   = 128,
        ppo_epochs:   int   = 4,
        clip_eps:     float = 0.2,
        entropy_coef: float = 0.03,
        value_coef:   float = 0.3,
        flat_threshold: float = 0.003,
        backtest_every: int = 50,
        sl_thresh:    float = 0.015,
        sl_penalty:   float = 0.002,
    ):
        print("\n=== Stage 2: PPO Fine-tuning ===")
        print(f"  rollout={rollout_size}  mini_batch={mini_batch}  "
              f"ppo_epochs={ppo_epochs}  n_updates={n_updates}")

        src    = feat_train if feat_train is not None else ohlcv_train
        close  = ohlcv_train[:, 3]
        N      = len(src)

        # pre-compute per-bar log returns and rolling std
        log_ret_full = np.zeros(N)
        for i in range(N - horizon):
            log_ret_full[i] = math.log(
                (close[i + horizon] + 1e-8) / (close[i] + 1e-8)
            )
        roll_std_full = np.zeros(N)
        for i in range(20, N):
            roll_std_full[i] = np.std(log_ret_full[max(0, i-20):i]) + 1e-6
        roll_std_full[:20] = roll_std_full[20] if N > 20 else 1e-4

        valid_idx = list(range(window, N - horizon))
        src_t     = torch.tensor(src, dtype=torch.float32)
        ppo_opt   = torch.optim.AdamW(
            self.model.parameters(), lr=5e-5, weight_decay=1e-4
        )

        best_sharpe = -float("inf")
        header = (f"  {'Upd':>5}  {'r_mean':>8}  {'r_std':>7}  "
                  f"{'L':>4} {'F':>4} {'S':>4}  "
                  f"{'conf':>5}  {'entr':>5}  {'gnorm':>5}")
        print(header)

        for update in range(1, n_updates + 1):
            # ---- collect rollout ----
            self.model.eval()
            idxs    = np.random.choice(valid_idx, size=rollout_size, replace=True)
            obs     = torch.stack([src_t[i - window: i] for i in idxs]).to(self.device)
            lr_b    = torch.tensor([log_ret_full[i]  for i in idxs],
                                   dtype=torch.float32).to(self.device)
            std_b   = torch.tensor([roll_std_full[i] for i in idxs],
                                   dtype=torch.float32).to(self.device)

            with torch.no_grad():
                logits_old, conf_old, _ = self.model(obs)
                dist_old   = torch.distributions.Categorical(
                    F.softmax(logits_old, dim=-1)
                )
                actions    = dist_old.sample()
                lp_old     = dist_old.log_prob(actions)
                rewards    = compute_reward(
                    actions, lr_b, conf_old.squeeze(1), std_b,
                    sl_thresh=sl_thresh, sl_penalty=sl_penalty,
                )
                advantages = rewards - rewards.mean()
                advantages = advantages / (advantages.std() + 1e-8)

            # ---- PPO updates with mini-batches ----
            self.model.train()
            perm = torch.randperm(rollout_size)
            total_pl, total_gnorm, nb = 0.0, 0.0, 0

            for _ in range(ppo_epochs):
                for start in range(0, rollout_size, mini_batch):
                    idx_mb = perm[start: start + mini_batch]
                    ob_mb  = obs[idx_mb]
                    ac_mb  = actions[idx_mb]
                    lp_mb  = lp_old[idx_mb]
                    adv_mb = advantages[idx_mb]

                    logits_new, _, _ = self.model(ob_mb)
                    dist_new  = torch.distributions.Categorical(
                        F.softmax(logits_new, dim=-1)
                    )
                    lp_new    = dist_new.log_prob(ac_mb)
                    entropy   = dist_new.entropy().mean()

                    ratio     = torch.exp(lp_new - lp_mb.detach())
                    surr1     = ratio * adv_mb
                    surr2     = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb
                    pl        = -torch.min(surr1, surr2).mean()
                    loss      = pl - entropy_coef * entropy

                    ppo_opt.zero_grad()
                    loss.backward()
                    gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    ppo_opt.step()
                    total_pl    += pl.item(); total_gnorm += gnorm.item(); nb += 1

            r_mean = rewards.mean().item()
            r_std  = rewards.std().item()
            ad     = torch.bincount(actions, minlength=3)

            if update % 10 == 0 or update == 1:
                print(f"  {update:>5d}  {r_mean:>+8.5f}  {r_std:>7.5f}  "
                      f"{ad[0]:>4d} {ad[1]:>4d} {ad[2]:>4d}  "
                      f"{conf_old.mean().item():>5.3f}  "
                      f"{entropy.item():>5.3f}  "
                      f"{total_gnorm/nb:>5.3f}")

            # ---- backtest-in-the-loop ----
            if update % backtest_every == 0:
                self.model.eval()
                with torch.no_grad():
                    act_np = actions.cpu().numpy()
                cl_np  = np.array([close[i] for i in idxs])
                bt     = run_backtest(act_np, cl_np)
                sharpe = bt["sharpe"]
                print(f"  [Backtest u{update}] sharpe={sharpe:.3f}  "
                      f"max_dd={bt['max_dd']:.4f}  "
                      f"win={bt['win_rate']:.3f}  "
                      f"ret={bt['total_return']:.4f}")
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    self.model.save(self.ckpt_dir / "best_ppo.pt")
            elif r_mean > (best_sharpe if best_sharpe > -float("inf") else -1):
                self.model.save(self.ckpt_dir / "best_ppo.pt")

            # free GPU memory each update
            del obs, lr_b, std_b, rewards, advantages, actions, lp_old
            if self.device != "cpu":
                torch.cuda.empty_cache()

        print(f"  Best backtest Sharpe: {best_sharpe:.4f}")
        print(f"  Checkpoint: {self.ckpt_dir}/best_ppo.pt")


# ---------------------------------------------------------------------------
# Ensemble training  (OOM-safe: sequential, not parallel)
# ---------------------------------------------------------------------------
def train_ensemble(
    model_cfg:     dict,
    X_train:       torch.Tensor,
    y_train:       torch.Tensor,
    X_val:         torch.Tensor,
    y_val:         torch.Tensor,
    ohlcv_train:   np.ndarray,
    feat_train:    Optional[np.ndarray],
    r_train:       Optional[torch.Tensor],
    r_val:         Optional[torch.Tensor],
    n_seeds:       int   = 3,
    device:        str   = "cpu",
    checkpoint_dir: str  = "checkpoints",
    pretrain_epochs: int = 30,
    ppo_updates:   int   = 500,
    batch_size:    int   = 128,
) -> List[Path]:
    """
    Train n_seeds independent models sequentially (OOM-safe).
    Returns list of checkpoint paths for ensemble inference.
    """
    ckpt_paths = []
    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE SEED {seed+1}/{n_seeds}")
        print(f"{'='*60}")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        random.seed(seed * 42)

        model   = BTCActor(**model_cfg)
        ckpt_s  = str(Path(checkpoint_dir) / f"seed_{seed}")
        trainer = ActorTrainer(model, device=device, checkpoint_dir=ckpt_s)

        trainer.masked_pretrain(X_train, epochs=3)
        trainer.pretrain(
            X_train, y_train, X_val, y_val, r_train, r_val,
            epochs=pretrain_epochs, batch_size=batch_size,
            patience=8, warmup_epochs=1, label_smoothing=0.05,
        )
        trainer.ppo_finetune(
            ohlcv_train=ohlcv_train,
            feat_train=feat_train,
            n_updates=ppo_updates,
            batch_size=batch_size,
        )

        best_ckpt = Path(ckpt_s) / "best_ppo.pt"
        ckpt_paths.append(best_ckpt)

        # free model from memory before next seed
        del model, trainer
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    return ckpt_paths


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def ensemble_signal(
    ckpt_paths: List[Path],
    ohlcv_window: torch.Tensor,
    device: str = "cpu",
    mc_samples: int = 0,
) -> dict:
    """
    Average logits across ensemble members.
    """
    if ohlcv_window.dim() == 2:
        ohlcv_window = ohlcv_window.unsqueeze(0)
    all_probs = []
    for ckpt in ckpt_paths:
        m = BTCActor.load(ckpt, map_location=device)
        m.eval()
        logits, _, _ = m(ohlcv_window.to(device))
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        del m
        gc.collect()
    probs  = torch.stack(all_probs).mean(0)
    action = probs.argmax(-1).item()
    return {
        "action":      action,
        "action_str":  ["LONG", "FLAT", "SHORT"][action],
        "probs":       {n: probs[0, i].item() for i, n in enumerate(["LONG","FLAT","SHORT"])},
        "uncertainty": torch.stack(all_probs).std(0).mean().item(),
    }
