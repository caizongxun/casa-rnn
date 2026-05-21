"""
funding.py -- Binance futures market data fetcher

Fetches: funding rate, long/short ratio, open interest (OI)
All endpoints are public (no API key required).
Fallback: if fetch fails, returns zero-filled arrays.
"""

from __future__ import annotations
import time
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


BASE_FAPI  = "https://fapi.binance.com"
BASE_DAPI  = "https://dapi.binance.com"


def _get(url, params, retries=3, timeout=10):
    if not _HAS_REQUESTS:
        return None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [funding] fetch failed {url}: {e}")
                return None
            time.sleep(1)


def fetch_funding_rate(
    symbol: str = "BTCUSDT",
    start_ms: int = 0,
    end_ms:   int = 0,
    limit:    int = 1000,
) -> Optional[np.ndarray]:
    """
    Returns ndarray of shape (N, 2): [timestamp_ms, fundingRate]
    """
    params = {"symbol": symbol, "limit": limit}
    if start_ms: params["startTime"] = start_ms
    if end_ms:   params["endTime"]   = end_ms
    data = _get(f"{BASE_FAPI}/fapi/v1/fundingRate", params)
    if not data:
        return None
    arr = np.array([[int(d["fundingTime"]), float(d["fundingRate"])]
                    for d in data])
    return arr


def fetch_ls_ratio(
    symbol: str = "BTCUSDT",
    period: str = "1h",
    limit:  int = 500,
    start_ms: int = 0,
) -> Optional[np.ndarray]:
    """
    Global Long/Short account ratio.
    Returns (N, 2): [timestamp_ms, longShortRatio]
    """
    params = {"symbol": symbol, "period": period, "limit": limit}
    if start_ms: params["startTime"] = start_ms
    data = _get(
        f"{BASE_FAPI}/futures/data/globalLongShortAccountRatio", params
    )
    if not data:
        return None
    arr = np.array([[int(d["timestamp"]), float(d["longShortRatio"])]
                    for d in data])
    return arr


def fetch_open_interest(
    symbol: str = "BTCUSDT",
    period: str = "1h",
    limit:  int = 500,
    start_ms: int = 0,
) -> Optional[np.ndarray]:
    """
    Open Interest history.
    Returns (N, 2): [timestamp_ms, sumOpenInterest]
    """
    params = {"symbol": symbol, "period": period, "limit": limit}
    if start_ms: params["startTime"] = start_ms
    data = _get(
        f"{BASE_FAPI}/futures/data/openInterestHist", params
    )
    if not data:
        return None
    arr = np.array([[int(d["timestamp"]), float(d["sumOpenInterest"])]
                    for d in data])
    return arr


def align_to_ohlcv(
    ohlcv_ts: np.ndarray,     # (N,) unix ms timestamps
    src:      np.ndarray,     # (M, 2) [timestamp_ms, value]
    default:  float = 0.0,
) -> np.ndarray:
    """
    Forward-fill src values onto ohlcv_ts grid.
    Returns (N,) array.
    """
    if src is None or len(src) == 0:
        return np.full(len(ohlcv_ts), default, dtype=np.float32)
    out = np.full(len(ohlcv_ts), default, dtype=np.float32)
    j   = 0
    for i, ts in enumerate(ohlcv_ts):
        while j + 1 < len(src) and src[j + 1, 0] <= ts:
            j += 1
        if src[j, 0] <= ts:
            out[i] = src[j, 1]
    return out


def load_or_fetch_futures_features(
    symbol:   str,
    df_index_ms: np.ndarray,   # timestamps from the main OHLCV df
    cache_dir: Path,
) -> dict:
    """
    Load from cache or fetch from Binance.
    Returns dict of 1-D arrays (same length as df_index_ms):
        funding_rate, ls_ratio, open_interest
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(df_index_ms[0])
    end_ms   = int(df_index_ms[-1])
    result   = {}

    for key, fetch_fn, kwargs in [
        ("funding_rate", fetch_funding_rate,
         {"symbol": symbol, "start_ms": start_ms,
          "end_ms": end_ms, "limit": 1000}),
        ("ls_ratio",     fetch_ls_ratio,
         {"symbol": symbol, "start_ms": start_ms, "limit": 500}),
        ("open_interest", fetch_open_interest,
         {"symbol": symbol, "start_ms": start_ms, "limit": 500}),
    ]:
        cache_f = cache_dir / f"{symbol}_{key}.npy"
        if cache_f.exists():
            raw = np.load(cache_f)
            print(f"  [funding] loaded cache {cache_f.name}  ({len(raw)} rows)")
        else:
            print(f"  [funding] fetching {key} from Binance...")
            raw = fetch_fn(**kwargs)
            if raw is not None and len(raw):
                np.save(cache_f, raw)
                print(f"  [funding] saved {cache_f.name}")
            else:
                raw = None
                print(f"  [funding] fallback zeros for {key}")
        result[key] = align_to_ohlcv(df_index_ms, raw)

    return result
