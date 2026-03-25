"""Unit tests for market structure analysis."""

import pandas as pd
import numpy as np
import pytest

from bot.structure import detect_swings, determine_trend, Trend, detect_market_shift_bull


def _make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """Helper: build a minimal OHLCV DataFrame."""
    n = len(highs)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs,
        "low": lows,
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        "tick_volume": [100] * n,
    })


class TestSwingDetection:
    def test_detects_obvious_swing_high(self):
        # A clear peak in the middle
        highs = [1.0, 1.1, 1.2, 1.3, 1.5, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0]
        lows = [0.9, 1.0, 1.1, 1.2, 1.4, 1.2, 1.1, 1.0, 0.9, 0.9, 0.9]
        df = _make_df(highs, lows)
        swings = detect_swings(df, period=2)
        swing_highs = [s for s in swings if s.is_high]
        assert any(s.price == 1.5 for s in swing_highs)

    def test_detects_obvious_swing_low(self):
        lows = [1.5, 1.4, 1.3, 1.2, 1.0, 1.2, 1.3, 1.4, 1.5, 1.5, 1.5]
        highs = [1.6, 1.5, 1.4, 1.3, 1.1, 1.3, 1.4, 1.5, 1.6, 1.6, 1.6]
        df = _make_df(highs, lows)
        swings = detect_swings(df, period=2)
        swing_lows = [s for s in swings if not s.is_high]
        assert any(s.price == 1.0 for s in swing_lows)


class TestTrend:
    def test_uptrend(self):
        from bot.structure import SwingPoint
        # Explicit HH/HL sequence
        swings = [
            SwingPoint(0, 1.10, pd.Timestamp("2025-01-01"), False),   # SL
            SwingPoint(5, 1.20, pd.Timestamp("2025-01-02"), True),    # SH
            SwingPoint(10, 1.15, pd.Timestamp("2025-01-03"), False),  # HL
            SwingPoint(15, 1.25, pd.Timestamp("2025-01-04"), True),   # HH
        ]
        assert determine_trend(swings) == Trend.UP

    def test_downtrend(self):
        from bot.structure import SwingPoint
        swings = [
            SwingPoint(0, 1.25, pd.Timestamp("2025-01-01"), True),    # SH
            SwingPoint(5, 1.15, pd.Timestamp("2025-01-02"), False),   # SL
            SwingPoint(10, 1.20, pd.Timestamp("2025-01-03"), True),   # LH
            SwingPoint(15, 1.10, pd.Timestamp("2025-01-04"), False),  # LL
        ]
        assert determine_trend(swings) == Trend.DOWN


class TestMarketShift:
    def test_bullish_shift_detected(self):
        from bot.structure import SwingPoint
        # Simulated LTF swing highs: SH1 > SH2 (lower high) < SH3 (break)
        swings = [
            SwingPoint(0, 1.200, pd.Timestamp("2025-01-01 00:00"), True),
            SwingPoint(5, 1.180, pd.Timestamp("2025-01-01 01:00"), True),   # LH
            SwingPoint(10, 1.195, pd.Timestamp("2025-01-01 02:00"), True),  # breaks LH
        ]
        result = detect_market_shift_bull(swings)
        assert result is not None
        assert result.price == 1.180

    def test_no_shift_when_still_trending_down(self):
        from bot.structure import SwingPoint
        swings = [
            SwingPoint(0, 1.200, pd.Timestamp("2025-01-01 00:00"), True),
            SwingPoint(5, 1.180, pd.Timestamp("2025-01-01 01:00"), True),
            SwingPoint(10, 1.170, pd.Timestamp("2025-01-01 02:00"), True),  # still LH
        ]
        result = detect_market_shift_bull(swings)
        assert result is None
