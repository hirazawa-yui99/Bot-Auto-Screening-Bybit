"""
analysis/technicals.py
Standalone technical indicator helpers (used by confidence scorer).
"""
from typing import Dict

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / (volume.rolling(window).mean() + 1e-12)


def stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3):
    r = rsi(series, period)
    min_r = r.rolling(period).min()
    max_r = r.rolling(period).max()
    stoch = (r - min_r) / (max_r - min_r + 1e-12) * 100
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / (atr_ + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / (atr_ + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.rolling(period).mean()


def compute_all(df: pd.DataFrame) -> Dict[str, float]:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    return {
        "ema_9": float(ema(close, 9).iloc[-1]),
        "ema_21": float(ema(close, 21).iloc[-1]),
        "ema_50": float(ema(close, 50).iloc[-1]),
        "ema_200": float(ema(close, 200).iloc[-1]) if len(close) >= 200 else float(close.iloc[-1]),
        "rsi": float(rsi(close).iloc[-1]),
        "atr": float(atr(high, low, close).iloc[-1]),
        "rvol": float(rvol(vol).iloc[-1]),
        "adx": float(adx(high, low, close).iloc[-1]),
    }