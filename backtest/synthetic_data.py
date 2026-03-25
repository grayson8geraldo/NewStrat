"""Generate realistic synthetic Forex OHLCV data for backtesting.

Uses geometric Brownian motion with mean-reversion to create price action
that includes trends, pullbacks, and ranging periods — suitable for testing
the SMC strategy without external data sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_forex_data(
    symbol: str = "EURUSD",
    timeframe_minutes: int = 15,
    num_bars: int = 3000,
    start_price: float = 1.0800,
    volatility: float = 0.0003,
    trend_strength: float = 0.00002,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data mimicking Forex price action.

    Creates data with:
    - Trending phases (up and down)
    - Consolidation / ranging zones
    - Impulse moves with pullbacks
    - Realistic candle wicks

    Args:
        symbol: Pair name (used for scaling defaults for JPY pairs)
        timeframe_minutes: Bar size in minutes
        num_bars: Total number of bars to generate
        start_price: Opening price
        volatility: Per-bar volatility (std dev of returns)
        trend_strength: Drift coefficient for trending bias
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume
    """
    rng = np.random.default_rng(seed)

    # Adjust for JPY pairs
    if "JPY" in symbol:
        start_price = start_price if start_price > 100 else 150.0
        volatility *= 100
        trend_strength *= 100

    # Generate regime-switching: trend up, trend down, range
    regime_length = num_bars // 8
    regimes = []
    for _ in range(num_bars // regime_length + 1):
        regime = rng.choice(["up", "down", "range"], p=[0.4, 0.3, 0.3])
        regimes.extend([regime] * regime_length)
    regimes = regimes[:num_bars]

    # Generate returns with regime-dependent drift
    returns = np.zeros(num_bars)
    for i in range(num_bars):
        if regimes[i] == "up":
            drift = trend_strength
        elif regimes[i] == "down":
            drift = -trend_strength
        else:
            drift = 0.0

        # Occasional impulse moves (smart money push)
        impulse = 0.0
        if rng.random() < 0.02:  # 2% chance of impulse bar
            impulse = rng.choice([-1, 1]) * volatility * rng.uniform(3, 6)

        returns[i] = drift + volatility * rng.standard_normal() + impulse

    # Build close prices from returns
    closes = np.zeros(num_bars)
    closes[0] = start_price
    for i in range(1, num_bars):
        closes[i] = closes[i - 1] * (1 + returns[i])

    # Build OHLC from close
    opens = np.zeros(num_bars)
    highs = np.zeros(num_bars)
    lows = np.zeros(num_bars)

    opens[0] = start_price
    for i in range(1, num_bars):
        opens[i] = closes[i - 1]  # open = previous close

    for i in range(num_bars):
        body = abs(closes[i] - opens[i])
        # Random wicks (upper and lower)
        upper_wick = rng.exponential(body * 0.5 + volatility * start_price * 0.3)
        lower_wick = rng.exponential(body * 0.5 + volatility * start_price * 0.3)

        highs[i] = max(opens[i], closes[i]) + upper_wick
        lows[i] = min(opens[i], closes[i]) - lower_wick

    # Generate volume (higher during impulse moves)
    base_volume = rng.integers(500, 2000, size=num_bars).astype(float)
    for i in range(num_bars):
        if abs(returns[i]) > 2 * volatility:
            base_volume[i] *= rng.uniform(2, 5)

    # Build timestamps
    start_time = pd.Timestamp("2025-06-01 00:00:00")
    times = pd.date_range(start=start_time, periods=num_bars, freq=f"{timeframe_minutes}min")

    df = pd.DataFrame({
        "time": times,
        "open": np.round(opens, 5),
        "high": np.round(highs, 5),
        "low": np.round(lows, 5),
        "close": np.round(closes, 5),
        "tick_volume": base_volume.astype(int),
    })

    return df


def generate_mtf_data(
    symbol: str = "EURUSD",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate aligned HTF (4H), MTF (1H), LTF (15m) datasets.

    Creates a consistent dataset where all timeframes are derived from the
    same underlying 15-minute data via resampling.

    Returns:
        (htf_df, mtf_df, ltf_df)
    """
    # Generate 15-minute data (LTF) — ~3000 bars ≈ 31 days
    ltf_df = generate_forex_data(
        symbol=symbol,
        timeframe_minutes=15,
        num_bars=3000,
        seed=seed,
    )

    # Resample to 1H (MTF)
    ltf_df_indexed = ltf_df.set_index("time")
    mtf_df = ltf_df_indexed.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    }).dropna().reset_index()

    # Resample to 4H (HTF)
    htf_df = ltf_df_indexed.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    }).dropna().reset_index()

    return htf_df, mtf_df, ltf_df
