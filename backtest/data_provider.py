"""Data provider abstraction — allows swapping MT5 for historical CSV/yfinance data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# yfinance ticker mapping for major Forex pairs (via Yahoo Finance)
# Yahoo uses "EURUSD=X" format
YAHOO_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}

# yfinance interval ↔ our timeframe mapping
INTERVAL_MAP = {
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": None,  # yfinance doesn't support 4h directly — we resample from 1h
    "D1": "1d",
    "W1": "1wk",
}

# yfinance limits: intraday data only available for the last 60 days
# For longer periods, daily data is used
MAX_INTRADAY_DAYS = 59


def download_yahoo(
    symbol: str,
    timeframe: str,
    period: str = "60d",
) -> pd.DataFrame:
    """Download Forex data from Yahoo Finance.

    Args:
        symbol: Forex pair name, e.g. "EURUSD"
        timeframe: One of M5, M15, M30, H1, H4, D1, W1
        period: yfinance period string, e.g. "60d", "6mo", "1y"

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume
    """
    ticker = YAHOO_SYMBOL_MAP.get(symbol, f"{symbol}=X")

    if timeframe == "H4":
        # Download 1H and resample to 4H
        interval = "1h"
    else:
        interval = INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    logger.info("Downloading %s %s (period=%s) from Yahoo Finance…", symbol, timeframe, period)
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        raise RuntimeError(f"No data returned for {ticker} interval={interval} period={period}")

    # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "tick_volume",
    })
    df = df[["open", "high", "low", "close", "tick_volume"]].copy()

    # Resample to 4H if needed
    if timeframe == "H4":
        df = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }).dropna()

    df = df.reset_index()
    # Standardise the time column name
    time_col = [c for c in df.columns if c.lower() in ("date", "datetime", "index")][0] if "time" not in df.columns else "time"
    if time_col != "time":
        df = df.rename(columns={time_col: "time"})

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    logger.info("Downloaded %d bars for %s %s", len(df), symbol, timeframe)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expected columns: time (or date/datetime), open, high, low, close, volume (or tick_volume)
    """
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = df.columns.str.lower().str.strip()
    rename_map = {}
    for col in df.columns:
        if col in ("date", "datetime"):
            rename_map[col] = "time"
        if col == "volume":
            rename_map[col] = "tick_volume"
    df = df.rename(columns=rename_map)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df
