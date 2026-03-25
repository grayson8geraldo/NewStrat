"""Market structure analysis — swing detection, trend identification, BOS/CHoCH."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Trend(Enum):
    UP = "uptrend"
    DOWN = "downtrend"
    RANGE = "ranging"


@dataclass
class SwingPoint:
    index: int          # bar index inside the DataFrame
    price: float
    time: pd.Timestamp
    is_high: bool       # True = swing high, False = swing low


# ---------------------------------------------------------------------------
# Swing detection (Williams fractals approach)
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, period: int = 5) -> list[SwingPoint]:
    """Return an ordered list of swing highs and lows.

    A swing high is a bar whose *high* is the highest of the surrounding
    ``2 * period + 1`` bars.  Swing lows are the mirror image on *low*.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swings: list[SwingPoint] = []

    for i in range(period, n - period):
        # Swing High
        if highs[i] == max(highs[i - period: i + period + 1]):
            swings.append(SwingPoint(i, highs[i], df["time"].iloc[i], is_high=True))
        # Swing Low
        if lows[i] == min(lows[i - period: i + period + 1]):
            swings.append(SwingPoint(i, lows[i], df["time"].iloc[i], is_high=False))

    # Sort by bar index (keeps chronological order)
    swings.sort(key=lambda s: s.index)
    return swings


# ---------------------------------------------------------------------------
# Trend via Higher-Highs / Higher-Lows  (or LL / LH for downtrend)
# ---------------------------------------------------------------------------

def determine_trend(swings: list[SwingPoint], min_points: int = 4) -> Trend:
    """Evaluate the most recent swing sequence and classify the trend.

    Needs at least *min_points* swing points (alternating HH/HL or LL/LH).
    """
    # Separate last swing highs and lows
    swing_highs = [s for s in swings if s.is_high]
    swing_lows = [s for s in swings if not s.is_high]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return Trend.RANGE

    # Check the most recent two of each
    hh = swing_highs[-1].price > swing_highs[-2].price  # higher high
    hl = swing_lows[-1].price > swing_lows[-2].price     # higher low

    ll = swing_lows[-1].price < swing_lows[-2].price     # lower low
    lh = swing_highs[-1].price < swing_highs[-2].price   # lower high

    if hh and hl:
        return Trend.UP
    if ll and lh:
        return Trend.DOWN
    return Trend.RANGE


# ---------------------------------------------------------------------------
# Market Shift detection on the LTF
# ---------------------------------------------------------------------------

def detect_market_shift_bull(swings: list[SwingPoint]) -> SwingPoint | None:
    """Return the swing-high that was broken to the upside (bullish CHoCH).

    In a local down-move the LTF makes LH/LL.  A bullish market shift
    happens when price breaks above the most recent Lower High.
    We check the last 3 swing highs: if the latest is higher than the
    previous one (which was lower than the one before), we have a shift.
    """
    swing_highs = [s for s in swings if s.is_high]
    if len(swing_highs) < 3:
        return None

    sh1, sh2, sh3 = swing_highs[-3], swing_highs[-2], swing_highs[-1]
    # sh2 is a Lower High relative to sh1, and sh3 breaks above sh2
    if sh2.price < sh1.price and sh3.price > sh2.price:
        return sh2  # the broken level
    return None


def detect_market_shift_bear(swings: list[SwingPoint]) -> SwingPoint | None:
    """Mirror of bullish shift — bearish CHoCH."""
    swing_lows = [s for s in swings if not s.is_high]
    if len(swing_lows) < 3:
        return None

    sl1, sl2, sl3 = swing_lows[-3], swing_lows[-2], swing_lows[-1]
    if sl2.price > sl1.price and sl3.price < sl2.price:
        return sl2
    return None
