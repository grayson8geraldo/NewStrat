"""Real-time market data feed via yfinance.

Provides live OHLCV bars for Forex pairs and futures without a broker account.
yfinance uses Yahoo Finance data — free, no API key required.

Supported instruments:
  Forex:    EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, etc.
  Futures:  ES (S&P 500 E-mini), NQ (Nasdaq), YM (Dow), GC (Gold), CL (Oil)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---- Symbol mapping to Yahoo Finance tickers ----

_FOREX_MAP = {
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
    "AUDJPY": "AUDJPY=X",
    "CHFJPY": "CHFJPY=X",
}

_FUTURES_MAP = {
    "ES": "ES=F",       # S&P 500 E-mini
    "NQ": "NQ=F",       # Nasdaq 100 E-mini
    "YM": "YM=F",       # Dow Jones E-mini
    "GC": "GC=F",       # Gold futures
    "CL": "CL=F",       # Crude Oil futures
    "SI": "SI=F",       # Silver futures
    "SPX": "^GSPC",     # S&P 500 index (for reference)
}

# Merge all known symbols
SYMBOL_MAP = {**_FOREX_MAP, **_FUTURES_MAP}

# Yahoo interval ↔ our timeframe
_INTERVAL_MAP = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "D1": "1d", "W1": "1wk",
}

# ---- Symbol metadata ----

SYMBOL_META: dict[str, dict] = {}

# Forex pairs
for pair in _FOREX_MAP:
    is_jpy = "JPY" in pair
    SYMBOL_META[pair] = {
        "type": "forex",
        "point": 0.001 if is_jpy else 0.00001,
        "digits": 3 if is_jpy else 5,
        "pip_size": 0.01 if is_jpy else 0.0001,
        "tick_value": 8.0 if is_jpy else 10.0,  # approx $ per pip per standard lot
        "volume_step": 0.01,
        "volume_min": 0.01,   # micro lots (0.01 = 1000 units)
        "volume_max": 100.0,
        "spread_pips": 1.5 if is_jpy else 1.2,  # typical spread
    }

# Futures
SYMBOL_META["ES"] = {
    "type": "futures",
    "point": 0.25,       # tick size
    "digits": 2,
    "pip_size": 0.25,
    "tick_value": 12.50,  # $12.50 per tick per contract
    "volume_step": 1,
    "volume_min": 1,
    "volume_max": 50,
    "spread_pips": 1.0,   # 0.25 point spread = 1 tick
}
SYMBOL_META["NQ"] = {
    "type": "futures", "point": 0.25, "digits": 2,
    "pip_size": 0.25, "tick_value": 5.0,
    "volume_step": 1, "volume_min": 1, "volume_max": 50, "spread_pips": 1.0,
}


def resolve_ticker(symbol: str) -> str:
    """Convert our symbol name to a Yahoo Finance ticker."""
    return SYMBOL_MAP.get(symbol, symbol)


def get_meta(symbol: str) -> dict:
    """Return metadata for a symbol. Falls back to generic Forex defaults."""
    return SYMBOL_META.get(symbol, SYMBOL_META["EURUSD"])


def fetch_bars(
    symbol: str,
    timeframe: str,
    count: int = 500,
) -> pd.DataFrame:
    """Fetch the latest *count* OHLCV bars for *symbol* at *timeframe*.

    This calls Yahoo Finance every time — suitable for polling every 1-5 min.

    Args:
        symbol: Our symbol name (e.g. "EURUSD", "ES")
        timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"
        count: Approximate number of bars desired

    Returns:
        DataFrame with: time, open, high, low, close, tick_volume
    """
    ticker = resolve_ticker(symbol)
    need_resample_4h = (timeframe == "H4")
    interval = _INTERVAL_MAP.get("H1" if need_resample_4h else timeframe)

    if interval is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Choose period to get enough bars
    # yfinance intraday data limited to 60 days; for M1 only 7 days
    if interval == "1m":
        period = "7d"
    elif interval in ("5m", "15m", "30m"):
        period = "60d"
    elif interval == "1h":
        period = "60d" if not need_resample_4h else "60d"
    else:
        period = "2y"

    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        raise RuntimeError(f"No data for {ticker} ({timeframe})")

    # Flatten MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "tick_volume",
    })[["open", "high", "low", "close", "tick_volume"]].copy()

    if need_resample_4h:
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna()

    df = df.reset_index()
    time_col = [c for c in df.columns if c not in ("open", "high", "low", "close", "tick_volume")][0]
    if time_col != "time":
        df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Trim to requested count
    if len(df) > count:
        df = df.tail(count).reset_index(drop=True)

    return df


def get_current_price(symbol: str) -> float:
    """Get the latest available price for *symbol*."""
    ticker = resolve_ticker(symbol)
    t = yf.Ticker(ticker)
    # Try fast_info first, then history
    try:
        price = t.fast_info.get("lastPrice") or t.fast_info.get("last_price")
        if price and price > 0:
            return float(price)
    except Exception:
        pass

    hist = t.history(period="1d", interval="1m")
    if hist.empty:
        hist = t.history(period="5d", interval="1h")
    if hist.empty:
        raise RuntimeError(f"Cannot get current price for {symbol}")
    return float(hist["Close"].iloc[-1])
