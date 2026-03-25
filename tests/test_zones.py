"""Unit tests for demand / supply zone detection."""

import pandas as pd
import numpy as np

from bot.structure import SwingPoint
from bot.zones import find_demand_zones, Zone, nearest_supply_zone_above


def _make_df(highs, lows, n=50):
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs,
        "low": lows,
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        "tick_volume": [100] * n,
    })


class TestDemandZones:
    def test_finds_zone_on_higher_high_break(self):
        # Construct a sequence: rise -> pullback -> higher high
        n = 30
        base = (
            list(np.linspace(1.0, 1.1, 10))   # rise
            + list(np.linspace(1.1, 1.05, 5))  # pullback
            + list(np.linspace(1.05, 1.15, 10))  # impulse to HH
            + list(np.linspace(1.15, 1.12, 5))   # rest
        )
        highs = [x + 0.005 for x in base]
        lows = [x - 0.005 for x in base]
        df = _make_df(highs, lows, n)

        swing_highs = [
            SwingPoint(9, highs[9], df["time"].iloc[9], True),
            SwingPoint(24, highs[24], df["time"].iloc[24], True),  # HH
        ]
        swing_lows = [
            SwingPoint(14, lows[14], df["time"].iloc[14], False),
        ]

        zones = find_demand_zones(df, swing_highs, swing_lows)
        assert len(zones) >= 1
        assert zones[0].zone_type == "demand"


class TestNearestSupplyZone:
    def test_returns_closest_above(self):
        zones = [
            Zone(1.15, 1.14, pd.Timestamp("2025-01-01"), 0, "supply"),
            Zone(1.20, 1.19, pd.Timestamp("2025-01-02"), 1, "supply"),
        ]
        result = nearest_supply_zone_above(1.10, zones)
        assert result is not None
        assert result.low == 1.14

    def test_returns_none_when_no_zone_above(self):
        zones = [
            Zone(1.05, 1.04, pd.Timestamp("2025-01-01"), 0, "supply"),
        ]
        result = nearest_supply_zone_above(1.10, zones)
        assert result is None
