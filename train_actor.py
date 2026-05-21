"""
train_actor.py  (v3)

Changes v3
----------
* warmup_epochs=1  (was 3) -- shorter warmup stops early epoch L/F/S flipping
* label_smoothing=0.05 (was 0.1) -- sharper loss surface, faster convergence
* patience=8 -- give model more time before early stop
* ppo batch_size = pretrain batch_size * 2 for stable RL estimates
* --ppo-batch-size override added

Usage
-----
    python train_actor.py                          # CPU default (d_model=128)
    python train_actor.py --d-model 256 --n-layers 4 --n-heads 8  # GPU
    python train_actor.py --pretrain-only
    python train_actor.py --ppo-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))

from casa_rnn.market_data import load_and_enrich
from btc_actor import BTCActor, ActorTrainer
from btc_actor.trainer import build_windows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",        default="data")
    p.add_argument("--ckpt-dir",        default="checkpoints")
    p.add_argument("--window",          type=int,   default=168)
    p.add_argument("--horizon",         type=int,   default=4)
    p.add_argument("--flat-thresh",     type=float, default=0.003)
    p.add_argument("--d-model",         type=int,   default=128)
    p.add_argument("--n-layers",        type=int,   default=3)
    p.add_argument("--n-heads",         type=int,   default=4)
    p.add_argument("--patch-size",      type=int,   default=4)
    p.add_argument("--pretrain-epochs", type=int,   default=30)
    p.add_argument("--ppo-updates",     type=int,   default=500)
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--ppo-batch-size",  type=int,   default=0,
                   help="RL batch size (default: batch-size * 2)")
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--val-split",       type=float, default=0.15)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrain-only",   action="store_true")
    p.add_argument("--ppo-only",        action="store_true")
    return p.parse_args()


def sanity_check(model, X_train, device):
    print("\n[Sanity] checking gradient flow...")
    model.train()
    xb = X_train[:8].to(device)
    yb = torch.zeros(8, dtype=torch.long).to(device)
    logits, conf, _ = model(xb)
    loss = nn.CrossEntropyLoss()(logits, yb)
    loss.backward()

    total_norm = 0.0
    zero_grad_layers = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
        else:
            zero_grad_layers.append(name)

    total_norm = total_norm ** 0.5
    print(f"  Total grad norm : {total_norm:.4f}")
    if zero_grad_layers:
        print(f"  WARNING no grad : {zero_grad_layers[:5]}")
    else:
        print("  All parameters have gradients -- OK")

    with torch.no_grad():
        ctx = model.encoder(xb)
        print(f"  Encoder output  : mean={ctx.mean():.4f}  std={ctx.std():.4f}  "
              f"min={ctx.min():.4f}  max={ctx.max():.4f}")

    for p in model.parameters():
        p.grad = None
    print("")


def main():
    args = parse_args()
    ppo_bs = args.ppo_batch_size if args.ppo_batch_size > 0 else args.batch_size * 2

    print(f"Device : {args.device}")
    print(f"Model  : d_model={args.d_model}  n_layers={args.n_layers}  n_heads={args.n_heads}")

    df, _ = load_and_enrich(
        symbol="BTCUSDT",
        interval="1h",
        data_dir=Path(args.data_dir),
        corr_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        verbose=True,
    )

    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
    print(f"\nRaw OHLCV: {ohlcv.shape}")
    print(f"Volume stats: min={ohlcv[:,4].min():.0f}  "
          f"max={ohlcv[:,4].max():.0f}  "
          f"mean={ohlcv[:,4].mean():.0f}")

    X, y = build_windows(ohlcv, args.window, args.horizon, args.flat_thresh)
    print(f"Windows: {X.shape}  Labels: {y.shape}")
    lc = torch.bincount(y)
    print(f"Labels: LONG={lc[0]}  FLAT={lc[1]}  SHORT={lc[2]}")

    n_val            = int(len(X) * args.val_split)
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val,   y_val   = X[-n_val:], y[-n_val:]
    ohlcv_train      = ohlcv[:len(ohlcv) - int(len(ohlcv) * args.val_split)]

    model = BTCActor(
        in_channels=5,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    sanity_check(model, X_train, args.device)

    trainer = ActorTrainer(
        model=model, device=args.device, lr=args.lr, checkpoint_dir=args.ckpt_dir
    )

    if not args.ppo_only:
        trainer.pretrain(
            X_train, y_train, X_val, y_val,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            patience=8,
            warmup_epochs=1,          # 縮短 warmup，避免前幾個 epoch 亂飄
            label_smoothing=0.05,     # 降低 smoothing，loss 更乾淨
        )
    else:
        ckpt_path = Path(args.ckpt_dir) / "best_pretrain.pt"
        if ckpt_path.exists():
            model = BTCActor.load(ckpt_path, map_location=args.device)
            trainer.model = model
        else:
            trainer.pretrain(
                X_train, y_train, X_val, y_val,
                epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                patience=8,
                warmup_epochs=1,
                label_smoothing=0.05,
            )

    if not args.pretrain_only:
        trainer.ppo_finetune(
            ohlcv_train=ohlcv_train,
            window=args.window,
            horizon=args.horizon,
            n_updates=args.ppo_updates,
            batch_size=ppo_bs,
            flat_threshold=args.flat_thresh,
        )

    print("\nDone.")
    print("Live inference:")
    print("  from btc_actor import BTCActor")
    print("  actor = BTCActor.load('checkpoints/best_ppo.pt')")
    print("  signal = actor.get_signal(ohlcv_window)  # shape (168, 5)")


if __name__ == "__main__":
    main()
