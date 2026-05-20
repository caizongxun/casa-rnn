"""
market_data.py -- Auto-discovery, validation, and enrichment of market data.

Design goals:
  1. Scan a data directory for CSV/zip files in Binance Data Vision format.
  2. For each required correlated asset (ETH, SOL, BNB, etc.):
       a. Check if a local file exists.
       b. If not: try downloading from Binance public REST API (no key needed).
       c. If download fails: print a clear human-readable message with exact
          instructions on where to get the file.
  3. Build enriched feature columns from Binance Data Vision's raw columns:
       taker_buy_ratio, volume_delta, close_strength, vol_momentum,
       momentum_diff (fast-slow).
  4. Compute cross-asset correlation features when correlated data is available.

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
import os
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

# Correlated assets we'd like to use as auxiliary signals
DEFAULT_CORR_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# Data directory relative to cwd (can be overridden)
DEFAULT_DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# DataRegistry: scans local filesystem for available market data
# ---------------------------------------------------------------------------

class DataRegistry:
    """
    Scans a directory for Binance Data Vision CSV/zip files.

    Expected filename conventions (all accepted):
        BTCUSDT-1h-2024-01.csv
        BTCUSDT-1h-2024-01.zip
        BTCUSDT-1h-*.csv  (any date suffix)
        btcusdt_1h.csv    (legacy flat files)

    After init, self.files maps (symbol_upper, interval) -> List[Path].
    """

    # Binance Data Vision pattern: SYMBOL-INTERVAL-YYYY-MM[.csv|.zip]
    _BDV_RE = re.compile(
        r"(?P<symbol>[A-Za-z0-9]+)-(?P<interval>[0-9]+[mhd])-",
        re.IGNORECASE,
    )
    # Legacy flat file pattern: symbol_interval.csv
    _LEGACY_RE = re.compile(
        r"(?P<symbol>[A-Za-z0-9]+)[_-](?P<interval>[0-9]+[mhd])\.(csv|zip)$",
        re.IGNORECASE,
    )

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
            name = p.name
            m = self._BDV_RE.match(name)
            if m:
                key = (m.group("symbol").upper(), m.group("interval").lower())
                self.files.setdefault(key, []).append(p)
                continue
            m = self._LEGACY_RE.match(name)
            if m:
                key = (m.group("symbol").upper(), m.group("interval").lower())
                self.files.setdefault(key, []).append(p)

    def has(self, symbol: str, interval: str) -> bool:
        return (symbol.upper(), interval.lower()) in self.files

    def get_paths(self, symbol: str, interval: str) -> List[Path]:
        return self.files.get((symbol.upper(), interval.lower()), [])

    def summary(self) -> str:
        lines = [f"DataRegistry ({self.data_dir}):"]
        if not self.files:
            lines.append("  (empty)")
        for (sym, iv), paths in sorted(self.files.items()):
            lines.append(f"  {sym} {iv}: {len(paths)} file(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV loading (handles zip + bare csv, Binance Data Vision format)
# ---------------------------------------------------------------------------

def _read_bdv_csv(path: Path) -> pd.DataFrame:
    """Read one Binance Data Vision CSV or zip into a DataFrame."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV inside {path}")
            with zf.open(csv_names[0]) as f:
                raw = f.read()
        buf = io.StringIO(raw.decode("utf-8"))
    else:
        buf = path

    df = pd.read_csv(buf, header=None)

    # Accept files with or without a header row
    if df.iloc[0, 0] == "open_time":
        df = df.iloc[1:].reset_index(drop=True)

    # Assign standard column names if the file has >= 12 columns
    if df.shape[1] >= 12:
        df.columns = BINANCE_DV_COLS[:df.shape[1]]
    elif df.shape[1] >= 6:
        # Minimal OHLCV
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

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


def load_symbol(symbol: str, interval: str, registry: DataRegistry) -> Optional[pd.DataFrame]:
    """Load and concatenate all local files for a symbol+interval."""
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
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Binance public REST API fallback (no API key required)
# ---------------------------------------------------------------------------

def _download_klines_api(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    save_path: Path,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Download klines from Binance public REST API and save as CSV.
    Handles pagination automatically (500 candles per request).
    """
    try:
        import requests
    except ImportError:
        print("  [market_data] 'requests' not installed. Run: pip install requests")
        return None

    all_rows = []
    current_start = start_ms
    limit = 500

    print(f"  [market_data] Downloading {symbol} {interval} from Binance API ...", flush=True)

    while current_start < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
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
        time.sleep(0.1)  # polite rate limiting

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=BINANCE_DV_COLS)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, header=False)
    print(f"  [market_data] Saved {len(df)} candles to {save_path}")

    # Parse and return
    return _read_bdv_csv(save_path)


# ---------------------------------------------------------------------------
# Feature engineering: enriched columns from Binance Data Vision raw data
# ---------------------------------------------------------------------------

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment-proxy and momentum features to a Binance DV DataFrame.

    New columns added:
        taker_buy_ratio    -- active buy pressure (0-1)
        volume_delta_norm  -- normalised net buy/sell volume
        close_strength     -- candle close position in [low, high]
        vol_momentum       -- short vs long-term volume ratio
        momentum_5         -- 5-bar price momentum
        momentum_20        -- 20-bar price momentum
        momentum_diff      -- fast minus slow momentum
        log_return         -- if not present
        hl_range           -- (high-low)/close normalised range
        vol_ma20_ratio     -- volume / 20-bar mean volume
        rsi14              -- 14-bar RSI
        atr14              -- 14-bar ATR (normalised by close)
    """
    df = df.copy()

    # --- Sentiment proxies (from Binance DV columns) ---
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        vol_safe = df["volume"].clip(lower=1e-8)
        df["taker_buy_ratio"]   = (df["taker_buy_volume"] / vol_safe).clip(0, 1)
        df["volume_delta_norm"] = (2.0 * df["taker_buy_ratio"] - 1.0)  # range [-1, 1]
    else:
        df["taker_buy_ratio"]   = 0.5
        df["volume_delta_norm"] = 0.0

    # --- K-bar shape ---
    hl = (df["high"] - df["low"]).clip(lower=1e-8)
    df["close_strength"] = ((df["close"] - df["low"]) / hl).clip(0, 1)

    # --- Volume momentum ---
    vol_ma5  = df["volume"].rolling(5,  min_periods=1).mean()
    vol_ma20 = df["volume"].rolling(20, min_periods=1).mean().clip(lower=1e-8)
    df["vol_momentum"]   = (vol_ma5 / vol_ma20).clip(0, 5)
    df["vol_ma20_ratio"] = df["volume"] / vol_ma20

    # --- Price momentum ---
    df["momentum_5"]  = df["close"].pct_change(5).fillna(0)
    df["momentum_20"] = df["close"].pct_change(20).fillna(0)
    df["momentum_diff"] = df["momentum_5"] - df["momentum_20"]

    # --- Standard features ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1).clip(lower=1e-8)).fillna(0)
    df["hl_range"]   = (df["high"] - df["low"]) / df["close"].clip(lower=1e-8)

    # --- RSI 14 ---
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean().clip(lower=1e-8)
    df["rsi14"] = (100 - 100 / (1 + gain / loss)) / 100.0  # normalised to [0,1]

    # --- ATR 14 ---
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = (tr.rolling(14, min_periods=1).mean() / df["close"].clip(lower=1e-8))

    return df


# ---------------------------------------------------------------------------
# Cross-asset correlation features
# ---------------------------------------------------------------------------

def build_correlation_features(
    btc_df: pd.DataFrame,
    corr_dfs: Dict[str, pd.DataFrame],
    windows: List[int] = (24, 72, 168),
) -> pd.DataFrame:
    """
    Compute cross-asset features aligned to BTC timestamps.

    For each correlated asset:
        - rolling_corr_{symbol}_{window}h  : rolling price correlation with BTC
        - rel_momentum_{symbol}            : asset 5-bar momentum minus BTC momentum

    Args:
        btc_df    : enriched BTC DataFrame (must have open_time, log_return)
        corr_dfs  : {symbol: enriched DataFrame} for each correlated asset
        windows   : rolling windows in bars (default: 24h, 72h, 168h for 1h data)

    Returns:
        btc_df with additional columns appended.
    """
    result = btc_df.set_index("open_time").copy()

    for symbol, cdf in corr_dfs.items():
        sym_short = symbol.replace("USDT", "").lower()
        cdf_idx   = cdf.set_index("open_time")["log_return"].rename(f"lr_{sym_short}")
        result    = result.join(cdf_idx, how="left")
        result[f"lr_{sym_short}"] = result[f"lr_{sym_short}"].fillna(0)

        for w in windows:
            col = f"corr_{sym_short}_{w}h"
            result[col] = (
                result["log_return"]
                .rolling(w, min_periods=max(2, w // 4))
                .corr(result[f"lr_{sym_short}"])
                .fillna(0)
            )

        # Relative momentum: if corr asset has momentum_5, subtract BTC's
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
# High-level entry point
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
    One-stop function: discover -> load -> enrich -> cross-asset features.

    Returns:
        df        : enriched DataFrame ready for windowing
        feat_cols : list of feature column names to feed the model
    """
    data_dir = Path(data_dir)
    registry = DataRegistry(data_dir)

    if verbose:
        print(registry.summary())

    # --- Load primary symbol ---
    df = load_symbol(symbol, interval, registry)
    if df is None:
        _print_missing_instructions(symbol, interval, data_dir)
        raise FileNotFoundError(
            f"Primary data not found: {symbol} {interval} in {data_dir}"
        )

    # Filter date range
    if date_range:
        start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
        df = df[(df["open_time"] >= start) & (df["open_time"] <= end)].reset_index(drop=True)

    df = enrich_features(df)

    # --- Load correlated assets ---
    corr_dfs: Dict[str, pd.DataFrame] = {}
    for csym in corr_symbols:
        cdf = load_symbol(csym, interval, registry)
        if cdf is None:
            if auto_download:
                cdf = _try_download_corr(
                    csym, interval, df, data_dir, verbose
                )
            if cdf is None:
                if verbose:
                    _print_missing_instructions(csym, interval, data_dir, is_corr=True)
                continue  # skip this asset, not fatal
        corr_dfs[csym] = enrich_features(cdf)

    # --- Build cross-asset features ---
    if corr_dfs:
        df = build_correlation_features(df, corr_dfs)

    # --- Define feature columns ---
    base_feats = [
        "open", "high", "low", "close", "volume",
        "log_return", "hl_range", "vol_ma20_ratio", "rsi14", "atr14",
        # new sentiment + momentum features
        "taker_buy_ratio", "volume_delta_norm", "close_strength",
        "vol_momentum", "momentum_diff",
    ]
    corr_feats = [c for c in df.columns if c.startswith(("corr_", "rel_mom_"))]
    feat_cols  = [f for f in base_feats + corr_feats if f in df.columns]

    if verbose:
        print(f"\n[market_data] Primary: {symbol} {interval}  rows={len(df):,}")
        print(f"[market_data] Correlated assets loaded: {list(corr_dfs.keys())}")
        print(f"[market_data] Feature columns ({len(feat_cols)}): {feat_cols}")

    df = df.dropna(subset=["close", "log_return"]).reset_index(drop=True)
    return df, feat_cols


# ---------------------------------------------------------------------------
# Auto-download helper
# ---------------------------------------------------------------------------

def _try_download_corr(
    symbol: str,
    interval: str,
    btc_df: pd.DataFrame,
    data_dir: Path,
    verbose: bool,
) -> Optional[pd.DataFrame]:
    """Attempt to download a correlated symbol from Binance API."""
    try:
        start_ms = int(btc_df["open_time"].min().timestamp() * 1000)
        end_ms   = int(btc_df["open_time"].max().timestamp() * 1000)
    except Exception:
        return None

    save_path = data_dir / f"{symbol}-{interval}-api.csv"
    cdf = _download_klines_api(symbol, interval, start_ms, end_ms, save_path)
    return cdf


# ---------------------------------------------------------------------------
# Human-readable instructions
# ---------------------------------------------------------------------------

def _print_missing_instructions(symbol: str, interval: str, data_dir: Path, is_corr: bool = False):
    prefix = "  [market_data]" if is_corr else "[market_data]"
    severity = "WARNING" if is_corr else "ERROR"
    print(f"\n{'='*60}")
    print(f"{prefix} {severity}: {symbol} {interval} data not found")
    print(f"{'='*60}")
    print(f"Expected location: {data_dir.resolve()}/")
    print(f"")
    print(f"Option 1 - Binance Data Vision (recommended):")
    print(f"  1. Go to https://data.binance.vision/")
    print(f"  2. Navigate to: data/spot/monthly/klines/{symbol}/{interval}/")
    print(f"  3. Download all ZIP files you need")
    print(f"  4. Place them in: {data_dir.resolve()}/")
    print(f"  5. Filename format: {symbol}-{interval}-YYYY-MM.zip")
    print(f"")
    print(f"Option 2 - Auto download (Binance public API, no key needed):")
    print(f"  Set auto_download=True in load_and_enrich() call")
    print(f"  OR run: python -m casa_rnn.market_data --symbol {symbol} --interval {interval}")
    print(f"")
    print(f"Required CSV columns (Binance Data Vision standard):")
    for i, col in enumerate(BINANCE_DV_COLS):
        print(f"  col {i:2d}: {col}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI: python -m casa_rnn.market_data --symbol ETHUSDT --interval 1h
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download market data from Binance public API")
    parser.add_argument("--symbol",   default="ETHUSDT",  help="e.g. ETHUSDT")
    parser.add_argument("--interval", default="1h",       help="e.g. 1h, 4h, 1d")
    parser.add_argument("--start",    default="2020-01-01", help="start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None,       help="end date YYYY-MM-DD (default: now)")
    parser.add_argument("--data-dir", default="data",     help="output directory")
    args = parser.parse_args()

    start_ts = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ts   = int((pd.Timestamp(args.end, tz="UTC") if args.end
                    else pd.Timestamp.utcnow()).timestamp() * 1000)
    out_dir  = Path(args.data_dir)
    out_path = out_dir / f"{args.symbol}-{args.interval}-api.csv"

    result = _download_klines_api(
        args.symbol, args.interval, start_ts, end_ts, out_path
    )
    if result is not None:
        print(f"Downloaded {len(result):,} candles for {args.symbol} {args.interval}")
    else:
        print("Download failed. Check network connection.")
