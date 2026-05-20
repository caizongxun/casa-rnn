"""
market_data.py -- Auto-discovery, validation, and enrichment of market data.

Design goals:
  1. Scan a data directory for CSV/zip files in Binance Data Vision format.
  2. For ALL symbols (primary + correlated):
       a. Check if a local file exists.
       b. If not and auto_download=True: download from Binance public REST API
          (no API key needed). Default start date: 2020-01-01.
       c. If download fails: print clear human-readable instructions and raise
          (primary) or skip (correlated).
  3. Build enriched feature columns from Binance Data Vision raw columns.
  4. Compute cross-asset correlation features when correlated data available.

Binance Data Vision CSV columns (fixed order):
  0  open_time          ms timestamp
  1  open
  2  high
  3  low
  4  close
  5  volume
  6  close_time
  7  quote_volume
  8  count              number of trades
  9  taker_buy_volume
  10 taker_buy_quote_volume
  11 ignore
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

BINANCE_DV_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

DEFAULT_CORR_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEFAULT_DATA_DIR     = Path("data")
DEFAULT_START_DATE   = "2020-01-01"   # used when auto-downloading primary symbol


# ---------------------------------------------------------------------------
# DataRegistry
# ---------------------------------------------------------------------------

class DataRegistry:
    _BDV_RE    = re.compile(r"(?P<symbol>[A-Za-z0-9]+)-(?P<interval>[0-9]+[mhd])-", re.IGNORECASE)
    _LEGACY_RE = re.compile(r"(?P<symbol>[A-Za-z0-9]+)[_-](?P<interval>[0-9]+[mhd])\.(csv|zip)$", re.IGNORECASE)

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.files: Dict[Tuple[str, str], List[Path]] = {}
        self._scan()

    def _scan(self):
        if not self.data_dir.exists():
            return
        for p in sorted(self.data_dir.rglob("*")):
            if p.suffix not in (".csv", ".zip"):
                continue
            for pattern in (self._BDV_RE, self._LEGACY_RE):
                m = pattern.match(p.name)
                if m:
                    key = (m.group("symbol").upper(), m.group("interval").lower())
                    self.files.setdefault(key, []).append(p)
                    break

    def has(self, symbol: str, interval: str) -> bool:
        return (symbol.upper(), interval.lower()) in self.files

    def get_paths(self, symbol: str, interval: str) -> List[Path]:
        return self.files.get((symbol.upper(), interval.lower()), [])

    def register(self, symbol: str, interval: str, path: Path):
        """Add a newly downloaded file to the registry without re-scanning."""
        key = (symbol.upper(), interval.lower())
        self.files.setdefault(key, []).append(path)

    def summary(self) -> str:
        lines = [f"DataRegistry ({self.data_dir}):"]
        if not self.files:
            lines.append("  (empty — will auto-download)")
        for (sym, iv), paths in sorted(self.files.items()):
            lines.append(f"  {sym} {iv}: {len(paths)} file(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _read_bdv_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV inside {path}")
            with zf.open(csv_names[0]) as f:
                raw = f.read()
        buf = io.StringIO(raw.decode("utf-8"))
    else:
        buf = path  # type: ignore[assignment]

    df = pd.read_csv(buf, header=None)

    if str(df.iloc[0, 0]).lower() == "open_time":
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] >= 12:
        df.columns = BINANCE_DV_COLS[:df.shape[1]]
    elif df.shape[1] >= 6:
        df.columns = ["open_time", "open", "high", "low", "close", "volume"] + \
                     [f"_c{i}" for i in range(df.shape[1] - 6)]
    else:
        raise ValueError(f"Unexpected column count ({df.shape[1]}) in {path}")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["taker_buy_volume", "count", "quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def load_symbol(symbol: str, interval: str, registry: DataRegistry) -> Optional[pd.DataFrame]:
    paths = registry.get_paths(symbol, interval)
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            frames.append(_read_bdv_csv(p))
        except Exception as e:
            print(f"  [market_data] WARNING: could not read {p}: {e}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Binance public REST API download (no key required)
# ---------------------------------------------------------------------------

def _download_klines_api(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    save_path: Path,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    try:
        import requests
    except ImportError:
        print("  [market_data] 'requests' not installed. Run: pip install requests")
        return None

    all_rows: list = []
    current_start = start_ms
    limit = 500

    print(f"  [market_data] Downloading {symbol} {interval} "
          f"({pd.Timestamp(start_ms, unit='ms').date()} -> "
          f"{pd.Timestamp(end_ms,   unit='ms').date()}) ...", flush=True)

    while current_start < end_ms:
        params = dict(symbol=symbol.upper(), interval=interval,
                      startTime=current_start, endTime=end_ms, limit=limit)
        for attempt in range(max_retries):
            try:
                resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [market_data] API request failed: {e}")
                    return None
                time.sleep(2 ** attempt)

        if not data:
            break
        all_rows.extend(data)
        last_open_time = data[-1][0]
        if last_open_time >= end_ms or len(data) < limit:
            break
        current_start = last_open_time + 1
        time.sleep(0.1)

    if not all_rows:
        print(f"  [market_data] No data returned for {symbol} {interval}")
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=BINANCE_DV_COLS)
    df.to_csv(save_path, index=False, header=False)
    print(f"  [market_data] Saved {len(df):,} candles -> {save_path}")
    return _read_bdv_csv(save_path)


def _auto_download(
    symbol: str,
    interval: str,
    data_dir: Path,
    registry: DataRegistry,
    ref_df: Optional[pd.DataFrame] = None,   # align date range to primary symbol
) -> Optional[pd.DataFrame]:
    """Try to download symbol from Binance API and register the result."""
    if ref_df is not None:
        start_ms = int(ref_df["open_time"].min().timestamp() * 1000)
        end_ms   = int(ref_df["open_time"].max().timestamp() * 1000)
    else:
        start_ms = int(pd.Timestamp(DEFAULT_START_DATE, tz="UTC").timestamp() * 1000)
        end_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)

    save_path = data_dir / f"{symbol}-{interval}-api.csv"
    df = _download_klines_api(symbol, interval, start_ms, end_ms, save_path)
    if df is not None:
        registry.register(symbol, interval, save_path)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        vol_safe = df["volume"].clip(lower=1e-8)
        df["taker_buy_ratio"]   = (df["taker_buy_volume"] / vol_safe).clip(0, 1)
        df["volume_delta_norm"] = 2.0 * df["taker_buy_ratio"] - 1.0
    else:
        df["taker_buy_ratio"]   = 0.5
        df["volume_delta_norm"] = 0.0

    hl = (df["high"] - df["low"]).clip(lower=1e-8)
    df["close_strength"] = ((df["close"] - df["low"]) / hl).clip(0, 1)

    vol_ma5  = df["volume"].rolling(5,  min_periods=1).mean()
    vol_ma20 = df["volume"].rolling(20, min_periods=1).mean().clip(lower=1e-8)
    df["vol_momentum"]   = (vol_ma5 / vol_ma20).clip(0, 5)
    df["vol_ma20_ratio"] = df["volume"] / vol_ma20

    df["momentum_5"]    = df["close"].pct_change(5).fillna(0)
    df["momentum_20"]   = df["close"].pct_change(20).fillna(0)
    df["momentum_diff"] = df["momentum_5"] - df["momentum_20"]

    df["log_return"] = np.log(df["close"] / df["close"].shift(1).clip(lower=1e-8)).fillna(0)
    df["hl_range"]   = (df["high"] - df["low"]) / df["close"].clip(lower=1e-8)

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean().clip(lower=1e-8)
    df["rsi14"] = (100 - 100 / (1 + gain / loss)) / 100.0

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean() / df["close"].clip(lower=1e-8)

    return df


# ---------------------------------------------------------------------------
# Cross-asset correlation features
# ---------------------------------------------------------------------------

def build_correlation_features(
    btc_df: pd.DataFrame,
    corr_dfs: Dict[str, pd.DataFrame],
    windows: List[int] = (24, 72, 168),
) -> pd.DataFrame:
    result = btc_df.set_index("open_time").copy()

    for symbol, cdf in corr_dfs.items():
        sym_short = symbol.replace("USDT", "").lower()
        cdf_idx   = cdf.set_index("open_time")["log_return"].rename(f"lr_{sym_short}")
        result    = result.join(cdf_idx, how="left")
        result[f"lr_{sym_short}"] = result[f"lr_{sym_short}"].fillna(0)

        for w in windows:
            result[f"corr_{sym_short}_{w}h"] = (
                result["log_return"]
                .rolling(w, min_periods=max(2, w // 4))
                .corr(result[f"lr_{sym_short}"])
                .fillna(0)
            )

        if "momentum_5" in cdf.columns:
            mom_col = cdf.set_index("open_time")["momentum_5"].rename(f"mom_{sym_short}")
            result  = result.join(mom_col, how="left")
            result[f"rel_mom_{sym_short}"] = (
                result.get(f"mom_{sym_short}", 0) - result.get("momentum_5", 0)
            ).fillna(0)
            result.drop(columns=[f"mom_{sym_short}"], errors="ignore", inplace=True)

        result.drop(columns=[f"lr_{sym_short}"], errors="ignore", inplace=True)

    return result.reset_index()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_and_enrich(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    data_dir: Path = DEFAULT_DATA_DIR,
    corr_symbols: List[str] = DEFAULT_CORR_SYMBOLS,
    date_range: Optional[Tuple[str, str]] = None,
    auto_download: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Discover -> (auto-download if missing) -> load -> enrich -> cross-asset features.

    Primary symbol:
      - If local file found: load it.
      - If not found and auto_download=True: download from Binance API (2020-01-01 -> now).
      - If download fails: raise FileNotFoundError with full instructions.

    Correlated symbols:
      - Same logic, but failure is non-fatal (skipped with a warning).

    Returns:
        df        : enriched DataFrame
        feat_cols : feature column names to pass to the model
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    registry = DataRegistry(data_dir)

    if verbose:
        print(registry.summary())

    # --- Primary symbol ---
    df = load_symbol(symbol, interval, registry)
    if df is None:
        if auto_download:
            print(f"[market_data] {symbol} {interval} not found locally. Downloading from Binance API ...")
            df = _auto_download(symbol, interval, data_dir, registry)
        if df is None:
            _print_missing_instructions(symbol, interval, data_dir)
            raise FileNotFoundError(f"Primary data not found: {symbol} {interval} in {data_dir}")

    if date_range:
        s = pd.Timestamp(date_range[0], tz="UTC")
        e = pd.Timestamp(date_range[1], tz="UTC")
        df = df[(df["open_time"] >= s) & (df["open_time"] <= e)].reset_index(drop=True)

    df = enrich_features(df)

    # --- Correlated symbols ---
    corr_dfs: Dict[str, pd.DataFrame] = {}
    for csym in corr_symbols:
        cdf = load_symbol(csym, interval, registry)
        if cdf is None and auto_download:
            cdf = _auto_download(csym, interval, data_dir, registry, ref_df=df)
        if cdf is None:
            if verbose:
                print(f"  [market_data] WARNING: {csym} {interval} unavailable, skipping.")
            continue
        corr_dfs[csym] = enrich_features(cdf)

    if corr_dfs:
        df = build_correlation_features(df, corr_dfs)

    base_feats = [
        "open", "high", "low", "close", "volume",
        "log_return", "hl_range", "vol_ma20_ratio", "rsi14", "atr14",
        "taker_buy_ratio", "volume_delta_norm", "close_strength",
        "vol_momentum", "momentum_diff",
    ]
    corr_feats = [c for c in df.columns if c.startswith(("corr_", "rel_mom_"))]
    feat_cols  = [f for f in base_feats + corr_feats if f in df.columns]

    if verbose:
        print(f"\n[market_data] {symbol} {interval}  rows={len(df):,}")
        print(f"[market_data] Correlated: {list(corr_dfs.keys())}")
        print(f"[market_data] Features ({len(feat_cols)}): {feat_cols}")

    df = df.dropna(subset=["close", "log_return"]).reset_index(drop=True)
    return df, feat_cols


# ---------------------------------------------------------------------------
# Instructions helper
# ---------------------------------------------------------------------------

def _print_missing_instructions(symbol: str, interval: str, data_dir: Path, is_corr: bool = False):
    tag = "WARNING" if is_corr else "ERROR"
    print(f"\n{'='*60}")
    print(f"[market_data] {tag}: {symbol} {interval} data not found and auto-download failed")
    print(f"{'='*60}")
    print(f"Manual option - Binance Data Vision:")
    print(f"  1. https://data.binance.vision/")
    print(f"  2. data/spot/monthly/klines/{symbol}/{interval}/")
    print(f"  3. Place ZIP files in: {data_dir.resolve()}/")
    print(f"  4. Filename: {symbol}-{interval}-YYYY-MM.zip")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download market data from Binance public API")
    parser.add_argument("--symbol",   default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start",    default=DEFAULT_START_DATE)
    parser.add_argument("--end",      default=None)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms   = int((pd.Timestamp(args.end, tz="UTC") if args.end
                    else pd.Timestamp.utcnow()).timestamp() * 1000)
    out_dir  = Path(args.data_dir)
    out_path = out_dir / f"{args.symbol}-{args.interval}-api.csv"

    result = _download_klines_api(args.symbol, args.interval, start_ms, end_ms, out_path)
    if result is not None:
        print(f"Done: {len(result):,} candles for {args.symbol} {args.interval}")
