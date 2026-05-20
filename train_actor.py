"""
train_actor.py -- entry point for BTC-Actor two-stage training.

Usage
-----
    python train_actor.py                          # defaults
    python train_actor.py --data-dir data          # custom data dir
    python train_actor.py --pretrain-only          # skip PPO
    python train_actor.py --ppo-only               # skip pretrain (load best_pretrain.pt)
    python train_actor.py --window 168 --horizon 4 # 1-week lookback, 4h ahead

What it does
------------
1. Load raw BTC OHLCV via casa_rnn.market_data  (same data pipeline you already have)
2. Build rolling windows -- no hand-crafted features
3. Stage 1: supervised direction pre-training
4. Stage 2: PPO fine-tune on simulated PnL
5. Save best_ppo.pt  -- ready for live inference

Live inference (after training)
--------------------------------
    from btc_actor import BTCActor
    import torch

    actor = BTCActor.load("checkpoints/best_ppo.pt")
    # ohlcv: numpy array shape (168, 5) = last 168 hourly candles, columns [O,H,L,C,V]
    window = torch.tensor(ohlcv, dtype=torch.float32)
    signal = actor.get_signal(window)
    print(signal)
    # {'action': 0, 'action_str': 'LONG', 'confidence': 0.823, 'probs': {...}}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from casa_rnn.market_data import load_and_enrich
from btc_actor import BTCActor, ActorTrainer
from btc_actor.trainer import build_windows


def parse_args():
    p = argparse.ArgumentParser(description="Train BTC-Actor")
    p.add_argument("--data-dir",      default="data")
    p.add_argument("--ckpt-dir",      default="checkpoints")
    p.add_argument("--window",        type=int,   default=168,   help="lookback candles")
    p.add_argument("--horizon",       type=int,   default=4,     help="predict N candles ahead")
    p.add_argument("--flat-thresh",   type=float, default=0.003, help="flat zone threshold")
    p.add_argument("--d-model",       type=int,   default=256)
    p.add_argument("--n-layers",      type=int,   default=4)
    p.add_argument("--n-heads",       type=int,   default=8)
    p.add_argument("--patch-size",    type=int,   default=4)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--ppo-updates",   type=int,   default=500)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--val-split",     type=float, default=0.15,  help="validation fraction")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrain-only", action="store_true")
    p.add_argument("--ppo-only",      action="store_true")
    p.add_argument("--no-corr",       action="store_true",        help="disable corr symbols")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {args.device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    corr_symbols = [] if args.no_corr else None   # None = use defaults
    df, _ = load_and_enrich(
        symbol="BTCUSDT",
        interval="1h",
        data_dir=Path(args.data_dir),
        corr_symbols=corr_symbols if corr_symbols is not None else ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        verbose=True,
    )

    # Only OHLCV -- let the model discover everything else
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
    print(f"\nRaw OHLCV: {ohlcv.shape}  (only feeding 5 channels to the model)")

    # ------------------------------------------------------------------
    # 2. Build windows
    # ------------------------------------------------------------------
    X, y = build_windows(
        ohlcv,
        window=args.window,
        horizon=args.horizon,
        flat_threshold=args.flat_thresh,
    )
    print(f"Windows: {X.shape}  Labels: {y.shape}")
    label_counts = torch.bincount(y)
    print(f"Labels: LONG={label_counts[0]}  FLAT={label_counts[1]}  SHORT={label_counts[2]}")

    # Train/val split (chronological -- no shuffle across the split)
    n_val = int(len(X) * args.val_split)
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val,   y_val   = X[-n_val:],  y[-n_val:]
    ohlcv_train = ohlcv[:len(ohlcv) - int(len(ohlcv) * args.val_split)]

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = BTCActor(
        in_channels=5,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    trainer = ActorTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        checkpoint_dir=args.ckpt_dir,
    )

    # ------------------------------------------------------------------
    # 4. Stage 1: Supervised pre-training
    # ------------------------------------------------------------------
    if not args.ppo_only:
        trainer.pretrain(
            X_train, y_train,
            X_val,   y_val,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
        )
    else:
        # Load existing pretrain checkpoint
        ckpt_path = Path(args.ckpt_dir) / "best_pretrain.pt"
        if ckpt_path.exists():
            model = BTCActor.load(ckpt_path, map_location=args.device)
            trainer.model = model
            print(f"Loaded pretrain checkpoint: {ckpt_path}")
        else:
            print(f"WARNING: --ppo-only set but {ckpt_path} not found. Running pretrain first.")
            trainer.pretrain(
                X_train, y_train, X_val, y_val,
                epochs=args.pretrain_epochs, batch_size=args.batch_size,
            )

    # ------------------------------------------------------------------
    # 5. Stage 2: PPO fine-tuning
    # ------------------------------------------------------------------
    if not args.pretrain_only:
        trainer.ppo_finetune(
            ohlcv_train=ohlcv_train,
            window=args.window,
            horizon=args.horizon,
            n_updates=args.ppo_updates,
            batch_size=args.batch_size,
            flat_threshold=args.flat_thresh,
        )

    print("\nDone.")
    print(f"Live inference checkpoint: {args.ckpt_dir}/best_ppo.pt")
    print("")
    print("Usage:")
    print("  from btc_actor import BTCActor")
    print("  actor = BTCActor.load('checkpoints/best_ppo.pt')")
    print("  signal = actor.get_signal(ohlcv_window)  # shape (168, 5)")
    print("  print(signal)")


if __name__ == "__main__":
    main()
