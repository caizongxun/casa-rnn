"""
discover_strategies.py

Stand-alone strategy discovery script.  Run AFTER train_actor.py finishes.
Loads the ensemble checkpoints, runs PatternMiner + RuleMiner on the
validation set, and writes strategies_report.md + strategies_report.json.

Usage
-----
    python discover_strategies.py
    python discover_strategies.py --checkpoint-dir checkpoints --n-patterns 15 --n-rules 20
    python discover_strategies.py --no-futures          # skip futures feature fetch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from casa_rnn.market_data        import load_and_enrich
from casa_rnn.funding            import load_or_fetch_futures_features
from btc_actor.regime            import fit_regime_labels
from btc_actor.actor             import BTCActor
from btc_actor.trainer           import build_windows
from btc_actor.strategy_miner   import PatternMiner, RuleMiner
from btc_actor.report_writer    import write_strategy_report


def _device():
    if torch.cuda.is_available():  return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",         default="BTCUSDT")
    ap.add_argument("--interval",       default="1h")
    ap.add_argument("--data-dir",       default="data")
    ap.add_argument("--cache-dir",      default="data/futures_cache")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--window",         type=int,   default=168)
    ap.add_argument("--horizon",        type=int,   default=4)
    ap.add_argument("--d-model",        type=int,   default=128)
    ap.add_argument("--n-heads",        type=int,   default=4)
    ap.add_argument("--n-layers",       type=int,   default=3)
    ap.add_argument("--dropout",        type=float, default=0.1)
    ap.add_argument("--val-frac",       type=float, default=0.15)
    ap.add_argument("--n-clusters",     type=int,   default=12)
    ap.add_argument("--n-patterns",     type=int,   default=10)
    ap.add_argument("--n-rules",        type=int,   default=15)
    ap.add_argument("--tree-depth",     type=int,   default=4)
    ap.add_argument("--no-futures",     action="store_true")
    args = ap.parse_args()

    device = _device()
    print(f"[discover] device={device}")

    # ----------------------------------------------------------------
    # 1. Load data (same as train_actor.py)
    # ----------------------------------------------------------------
    df, feat_cols = load_and_enrich(
        symbol=args.symbol, interval=args.interval,
        data_dir=Path(args.data_dir), auto_download=True, verbose=False,
    )

    ts_ms = (df["open_time"].astype(np.int64) // 1_000_000).values

    if not args.no_futures:
        try:
            fut = load_or_fetch_futures_features(
                symbol=args.symbol, df_index_ms=ts_ms,
                cache_dir=Path(args.cache_dir),
            )
            for col in ["funding_rate", "ls_ratio", "open_interest"]:
                df[col] = fut[col].astype(np.float32)
            oi = df["open_interest"].values.copy()
            oi_mean = oi[oi > 0].mean() if (oi > 0).any() else 1.0
            df["open_interest"] = (oi / (oi_mean + 1e-8)).clip(0, 5).astype(np.float32)
            feat_cols = feat_cols + ["funding_rate", "ls_ratio", "open_interest"]
        except Exception:
            for col in ["funding_rate", "ls_ratio", "open_interest"]:
                df[col] = 0.0
            feat_cols = feat_cols + ["funding_rate", "ls_ratio", "open_interest"]

    feat_all  = df[feat_cols].values.astype(np.float32)
    ohlcv_all = df[["open","high","low","close","volume"]].values.astype(np.float32)
    atr_idx   = feat_cols.index("atr14") if "atr14" in feat_cols else 9

    N         = len(feat_all)
    val_start = int(N * (1 - args.val_frac))
    val_feat  = feat_all[val_start:]
    val_ohlcv = ohlcv_all[val_start:]
    close_all = ohlcv_all[:, 3]
    atr_all   = feat_all[:, atr_idx]
    regime_all = fit_regime_labels(close_all, atr_all, n_states=3)
    regime_val = regime_all[val_start:]

    X_val, y_val, _ = build_windows(
        ohlcv=val_ohlcv, window=args.window, horizon=args.horizon,
        feat_all=val_feat, atr_col_idx=atr_idx, regime_labels=regime_val,
    )
    in_channels = X_val.shape[-1]
    print(f"[discover] Val windows: {len(X_val):,}  features: {in_channels}")

    # ----------------------------------------------------------------
    # 2. Load model(s) — try seed_0/best_pretrain.pt first, then scan
    # ----------------------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_candidates = sorted(ckpt_dir.glob("seed_*/best_pretrain.pt")) + \
                      sorted(ckpt_dir.glob("best_pretrain.pt"))

    if not ckpt_candidates:
        print(f"[discover] ERROR: no checkpoint found in {ckpt_dir}")
        print("  Run train_actor.py first, then re-run this script.")
        sys.exit(1)

    ckpt_path = ckpt_candidates[0]
    print(f"[discover] Loading model from {ckpt_path}")

    model_cfg = dict(
        in_channels = in_channels,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
    )
    model = BTCActor(**model_cfg)
    state = torch.load(ckpt_path, map_location=device)
    # Support both bare state_dict and wrapped checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # ----------------------------------------------------------------
    # 3. Mine patterns
    # ----------------------------------------------------------------
    pm = PatternMiner(model, device=device, feature_names=feat_cols)
    patterns = pm.mine(
        X_val, val_ohlcv,
        horizon    = args.horizon,
        n_clusters = args.n_clusters,
    )

    # ----------------------------------------------------------------
    # 4. Mine rules
    # ----------------------------------------------------------------
    rm = RuleMiner(feature_names=feat_cols, max_depth=args.tree_depth)
    rules = rm.mine(
        X_val, y_val, model,
        device  = device,
        ohlcv   = val_ohlcv,
        horizon = args.horizon,
    )

    # ----------------------------------------------------------------
    # 5. Write report
    # ----------------------------------------------------------------
    paths = write_strategy_report(
        patterns        = patterns,
        rules           = rules,
        output_dir      = str(ckpt_dir),
        top_k_patterns  = args.n_patterns,
        top_k_rules     = args.n_rules,
        symbol          = args.symbol,
        interval        = args.interval,
    )
    print(f"\n[discover] Done.")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON:     {paths['json']}")
    print("\nOpen strategies_report.md to see what the model learned.")


if __name__ == "__main__":
    main()
