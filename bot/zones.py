"""Demand / Supply zone detection (Order Blocks) on the MTF."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from bot.structure import SwingPoint

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    high: float           # upper boundary
    low: float            # lower boundary
    time: pd.Timestamp    # time of the origin candle
    bar_index: int
    zone_type: str        # "demand" or "supply"
    mitigated: bool = False

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


# ---------------------------------------------------------------------------
# Pivot method — find the origin candle of the impulse move
# ---------------------------------------------------------------------------

def find_demand_zones(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> list[Zone]:
    """Detect demand zones (bullish order blocks) using the Pivot method.

    For each pair of consecutive swing highs where the later one is a
    Higher High (HH), we locate the lowest candle between the prior swing
    low and the HH — that candle's body defines the demand zone.
    """
    zones: list[Zone] = []

    for i in range(1, len(swing_highs)):
        prev_hh = swing_highs[i - 1]
        curr_hh = swing_highs[i]

        # Only interested in a Higher-High break
        if curr_hh.price <= prev_hh.price:
            continue

        # Find the swing low between the two highs
        between_lows = [
            s for s in swing_lows
            if prev_hh.index < s.index < curr_hh.index
        ]
        if not between_lows:
            continue

        deepest_low = min(between_lows, key=lambda s: s.price)

        # The origin candle: the bar with the lowest low between
        # that swing low and the start of the impulse up
        segment = df.iloc[deepest_low.index: curr_hh.index + 1]
        if segment.empty:
            continue

        origin_idx = segment["low"].idxmin()
        origin = df.iloc[origin_idx]

        zone = Zone(
            high=max(origin["open"], origin["close"]),
            low=origin["low"],
            time=origin["time"],
            bar_index=origin_idx,
            zone_type="demand",
        )
        zones.append(zone)

    return zones


def find_supply_zones(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> list[Zone]:
    """Detect supply zones (bearish order blocks) — mirror of demand."""
    zones: list[Zone] = []

    for i in range(1, len(swing_lows)):
        prev_ll = swing_lows[i - 1]
        curr_ll = swing_lows[i]

        if curr_ll.price >= prev_ll.price:
            continue

        between_highs = [
            s for s in swing_highs
            if prev_ll.index < s.index < curr_ll.index
        ]
        if not between_highs:
            continue

        highest_high = max(between_highs, key=lambda s: s.price)

        segment = df.iloc[highest_high.index: curr_ll.index + 1]
        if segment.empty:
            continue

        origin_idx = segment["high"].idxmax()
        origin = df.iloc[origin_idx]

        zone = Zone(
            high=origin["high"],
            low=min(origin["open"], origin["close"]),
            time=origin["time"],
            bar_index=origin_idx,
            zone_type="supply",
        )
        zones.append(zone)

    return zones


def nearest_supply_zone_above(price: float, zones: list[Zone]) -> Zone | None:
    """Return the closest un-mitigated supply zone above *price*."""
    candidates = [z for z in zones if z.zone_type == "supply" and z.low > price and not z.mitigated]
    if not candidates:
        return None
    return min(candidates, key=lambda z: z.low)
