"""
train_actor.py  v5

Changes vs v4
-------------
* Futures features (funding_rate, ls_ratio, open_interest) auto-fetched
  from Binance via funding.py; zero-filled fallback if network fails.
* in_channels auto-detected from actual feature matrix (was hardcoded).
* Walk-forward CV: n_folds=0 -> model itself explores optimal fold count
  via a lightweight grid search (3 candidate fold sizes, pick best mean acc).
* Ensemble 3 seeds: always sequential (OOM-safe); explicit gc + cuda empty
  between seeds.  Memory guard: if a seed OOMs, skip + warn, keep others.
* HMM regime labels: uses hmmlearn if installed, rule-based fallback.
  No change needed in regime.py (already handles this).
* flat_threshold: fully learnable / dynamic via ATR (in trainer.py).
  train_actor.py just passes atr_col_idx correctly.

Usage
-----
    python train_actor.py
    python train_actor.py --seeds 1 --no-wfcv
    python train_actor.py --pretrain-only
    python train_actor.py --ppo-only
    python train_actor.py --d-model 256 --n-layers 4 --n-heads 8
    python train_actor.py --no-futures   # skip funding/ls/oi fetch
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from casa_rnn.market_data   import load_and_enrich
from casa_rnn.funding       import load_or_fetch_futures_features
from btc_actor.regime       import fit_regime_labels
from btc_actor.actor        import BTCActor
from btc_actor.trainer      import (
    ActorTrainer, build_windows,
    train_ensemble, ensemble_signal,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_optimal_folds(
    X: torch.Tensor,
    y: torch.Tensor,
    r,
    device: str,
    model_cfg: dict,
    min_train_frac: float = 0.5,
    candidates: tuple = (6, 9, 12),
    epochs_per_fold: int = 8,
    batch_size: int = 128,
) -> int:
    """
    Light grid search over n_folds candidates.
    Trains a *small* model for epochs_per_fold epochs on each candidate fold
    count and returns the n_folds with highest mean val_acc.

    This is the "let the model find its own rhythm" part (#18 / #26):
    the walk-forward fold granularity is treated as a hyper-parameter
    searched at startup rather than fixed by the user.

    Only the first 3 folds of each candidate are evaluated (speed trade-off).
    """
    import copy
    print("\n[fold-search] Searching optimal walk-forward fold count ...")
    best_n, best_acc = candidates[0], -1.0

    for n_folds in candidates:
        N = len(X)
        test_size = max(1, (N - int(N * min_train_frac)) // n_folds)
        accs = []
        for fold in range(min(n_folds, 3)):   # quick check: only first 3 folds
            train_end = int(N * min_train_frac) + fold * test_size
            test_end  = min(train_end + test_size, N)
            if test_end <= train_end:
                break
            Xt, yt = X[:train_end], y[:train_end]
            Xv, yv = X[train_end:test_end], y[train_end:test_end]
            rt = r[:train_end]          if r is not None else None
            rv = r[train_end:test_end]  if r is not None else None

            probe = BTCActor(**model_cfg)
            tr    = ActorTrainer(
                probe, device=device,
                checkpoint_dir=f"/tmp/fold_probe_{n_folds}_{fold}",
            )
            try:
                tr.pretrain(Xt, yt, Xv, yv, rt, rv,
                            epochs=epochs_per_fold, batch_size=batch_size,
                            patience=3, warmup_epochs=1, label_smoothing=0.05)
                accs.append(tr.best_val_acc)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [fold-search] OOM at n_folds={n_folds} fold={fold}, skip")
                else:
                    raise
            finally:
                del probe, tr
                gc.collect()
                if device != "cpu":
                    torch.cuda.empty_cache()

        mean_acc = float(np.mean(accs)) if accs else 0.0
        print(f"  n_folds={n_folds:>3d}  mean_acc(first3)={mean_acc:.4f}")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_n   = n_folds

    print(f"[fold-search] -> optimal n_folds={best_n}  (acc={best_acc:.4f})\n")
    return best_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",          default="BTCUSDT")
    ap.add_argument("--interval",        default="1h")
    ap.add_argument("--data-dir",        default="data")
    ap.add_argument("--cache-dir",       default="data/futures_cache")
    ap.add_argument("--window",          type=int,   default=168)
    ap.add_argument("--horizon",         type=int,   default=4)
    ap.add_argument("--d-model",         type=int,   default=128)
    ap.add_argument("--n-heads",         type=int,   default=4)
    ap.add_argument("--n-layers",        type=int,   default=3)
    ap.add_argument("--dropout",         type=float, default=0.1)
    ap.add_argument("--seeds",           type=int,   default=3)
    ap.add_argument("--pretrain-epochs", type=int,   default=30)
    ap.add_argument("--ppo-updates",     type=int,   default=500)
    ap.add_argument("--batch-size",      type=int,   default=128)
    ap.add_argument("--no-wfcv",         action="store_true")
    ap.add_argument("--no-futures",      action="store_true",
                    help="Skip fetching funding/ls/oi from Binance")
    ap.add_argument("--pretrain-only",   action="store_true")
    ap.add_argument("--ppo-only",        action="store_true")
    ap.add_argument("--checkpoint-dir",  default="checkpoints")
    ap.add_argument("--val-frac",        type=float, default=0.15)
    args = ap.parse_args()

    device = _device()
    print(f"[train_actor] device={device}  seeds={args.seeds}")

    # ------------------------------------------------------------------
    # 1.  Load OHLCV + base features
    #     load_and_enrich() calls _auto_download() which tries:
    #       a. Binance Data Vision (S3, no geo-block)
    #       b. REST API fallback
    # ------------------------------------------------------------------
    df, feat_cols = load_and_enrich(
        symbol       = args.symbol,
        interval     = args.interval,
        data_dir     = Path(args.data_dir),
        auto_download= True,
        verbose      = True,
    )
    print(f"[train_actor] Loaded {len(df):,} candles  base_feats={len(feat_cols)}")

    # ------------------------------------------------------------------
    # 2.  Futures features  (funding rate, long/short ratio, OI)
    #     load_or_fetch_futures_features() caches to disk and fills zeros
    #     on any network failure, so this is always safe.
    # ------------------------------------------------------------------
    ts_ms = (df["open_time"].astype(np.int64) // 1_000_000).values

    if not args.no_futures:
        print("[train_actor] Fetching futures features (funding/ls/oi) ...")
        try:
            fut = load_or_fetch_futures_features(
                symbol      = args.symbol,
                df_index_ms = ts_ms,
                cache_dir   = Path(args.cache_dir),
            )
            df["funding_rate"]  = fut["funding_rate"].astype(np.float32)
            df["ls_ratio"]      = fut["ls_ratio"].astype(np.float32)
            df["open_interest"] = fut["open_interest"].astype(np.float32)
            # Normalize OI to a returns-like scale (divide by mean)
            oi = df["open_interest"].values.copy()
            oi_mean = oi[oi > 0].mean() if (oi > 0).any() else 1.0
            df["open_interest"] = (oi / (oi_mean + 1e-8)).clip(0, 5).astype(np.float32)
            feat_cols = feat_cols + ["funding_rate", "ls_ratio", "open_interest"]
            print(f"[train_actor] Futures features appended. total_feats={len(feat_cols)}")
        except Exception as e:
            print(f"[train_actor] WARNING: futures fetch failed ({e}), using zeros")
            for col in ["funding_rate", "ls_ratio", "open_interest"]:
                df[col] = 0.0
            feat_cols = feat_cols + ["funding_rate", "ls_ratio", "open_interest"]
    else:
        print("[train_actor] --no-futures: skipping futures features")

    # ------------------------------------------------------------------
    # 3.  Build feature matrix  (in_channels auto-detected here)
    # ------------------------------------------------------------------
    feat_all   = df[feat_cols].values.astype(np.float32)    # (N, C)
    ohlcv_all  = df[["open","high","low","close","volume"]].values.astype(np.float32)

    # atr14 column index used by dynamic flat_threshold in trainer.py
    atr_col_idx = feat_cols.index("atr14") if "atr14" in feat_cols else 9

    N          = len(feat_all)
    val_start  = int(N * (1 - args.val_frac))
    train_feat  = feat_all[:val_start]
    val_feat    = feat_all[val_start:]
    train_ohlcv = ohlcv_all[:val_start]
    val_ohlcv   = ohlcv_all[val_start:]

    # ------------------------------------------------------------------
    # 4.  HMM regime labels
    #     fit_regime_labels() tries hmmlearn first; rule-based fallback.
    # ------------------------------------------------------------------
    print("[train_actor] Fitting regime labels ...")
    close_all  = ohlcv_all[:, 3]
    atr_all    = feat_all[:, atr_col_idx]
    regime_all = fit_regime_labels(close_all, atr_all, n_states=3)
    regime_train = regime_all[:val_start]
    regime_val   = regime_all[val_start:]

    # ------------------------------------------------------------------
    # 5.  Build windows  (dynamic flat_threshold via ATR, #18)
    # ------------------------------------------------------------------
    print("[train_actor] Building windows ...")
    X_train, y_train, r_train = build_windows(
        ohlcv         = train_ohlcv,
        window        = args.window,
        horizon       = args.horizon,
        feat_all      = train_feat,
        atr_col_idx   = atr_col_idx,
        regime_labels = regime_train,
    )
    X_val, y_val, r_val = build_windows(
        ohlcv         = val_ohlcv,
        window        = args.window,
        horizon       = args.horizon,
        feat_all      = val_feat,
        atr_col_idx   = atr_col_idx,
        regime_labels = regime_val,
    )
    print(f"[train_actor] Windows: train={len(X_train):,}  val={len(X_val):,}  "
          f"in_channels={X_train.shape[-1]}")

    in_channels = X_train.shape[-1]   # auto-detected, not hardcoded

    # ------------------------------------------------------------------
    # 6.  Model config
    # ------------------------------------------------------------------
    model_cfg = dict(
        in_channels = in_channels,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
    )

    # ------------------------------------------------------------------
    # 7.  Walk-forward CV with auto fold-count optimisation  (#26)
    #     Skipped when --no-wfcv or --ppo-only.
    # ------------------------------------------------------------------
    if not args.no_wfcv and not args.ppo_only:
        X_all_wf = torch.cat([X_train, X_val])
        y_all_wf = torch.cat([y_train, y_val])
        r_all_wf = torch.cat([r_train, r_val]) if r_train is not None else None

        # Let the model find its own optimal fold granularity
        optimal_folds = _find_optimal_folds(
            X_all_wf, y_all_wf, r_all_wf,
            device          = device,
            model_cfg       = model_cfg,
            min_train_frac  = 0.5,
            candidates      = (6, 9, 12),
            epochs_per_fold = 8,
            batch_size      = args.batch_size,
        )

        probe_model   = BTCActor(**model_cfg)
        probe_trainer = ActorTrainer(
            probe_model, device=device,
            checkpoint_dir=str(Path(args.checkpoint_dir) / "wfcv"),
        )
        probe_trainer.walk_forward_cv(
            X_all_wf, y_all_wf, r_all_wf,
            n_folds    = optimal_folds,
            epochs     = 15,
            batch_size = args.batch_size,
        )
        del probe_model, probe_trainer
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 8.  Ensemble training  (3 seeds sequential, OOM-safe)  (#27)
    #     train_ensemble() wraps each seed in try/except so one OOM
    #     does not abort the remaining seeds.
    # ------------------------------------------------------------------
    if not args.pretrain_only:
        ckpt_paths = train_ensemble(
            model_cfg       = model_cfg,
            X_train         = X_train,
            y_train         = y_train,
            X_val           = X_val,
            y_val           = y_val,
            ohlcv_train     = train_ohlcv,
            feat_train      = train_feat,
            r_train         = r_train,
            r_val           = r_val,
            n_seeds         = args.seeds,
            device          = device,
            checkpoint_dir  = args.checkpoint_dir,
            pretrain_epochs = args.pretrain_epochs,
            ppo_updates     = args.ppo_updates,
            batch_size      = args.batch_size,
        )
        print(f"\n[train_actor] Ensemble checkpoints:")
        for p in ckpt_paths:
            print(f"  {p}")
    else:
        # --pretrain-only: stage 0 + stage 1 only, single model
        model   = BTCActor(**model_cfg)
        trainer = ActorTrainer(model, device=device,
                               checkpoint_dir=args.checkpoint_dir)
        trainer.masked_pretrain(X_train, epochs=5)
        trainer.pretrain(X_train, y_train, X_val, y_val, r_train, r_val,
                         epochs=args.pretrain_epochs,
                         batch_size=args.batch_size)
        ckpt_paths = [Path(args.checkpoint_dir) / "best_pretrain.pt"]
        del model, trainer
        gc.collect()

    # ------------------------------------------------------------------
    # 9.  Sanity inference: ensemble signal on last val window
    # ------------------------------------------------------------------
    if ckpt_paths:
        sample_window = X_val[:1]
        sig = ensemble_signal(ckpt_paths, sample_window, device=device)
        print(f"\n[train_actor] Sample signal: {sig['action_str']}  "
              f"probs={sig['probs']}  uncertainty={sig['uncertainty']:.4f}")

    print("\n[train_actor] Done.")


if __name__ == "__main__":
    main()
