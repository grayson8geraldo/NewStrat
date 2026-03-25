"""Position sizing and risk management."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def pip_value(symbol_info) -> float:
    """Return the monetary value of 1 pip for a standard lot (1.0).

    For most Forex pairs 1 pip = 0.0001; for JPY pairs 1 pip = 0.01.
    ``symbol_info.trade_tick_value`` gives the profit per tick for 1 lot.
    """
    tick_size = symbol_info.trade_tick_size  # e.g. 0.00001
    point = symbol_info.point               # smallest price increment
    tick_value = symbol_info.trade_tick_value

    # pip = 10 points for 5-digit brokers, 1 point for 4-digit
    pips_per_tick = tick_size / point  # usually 1
    # value of 1 pip = tick_value * (pip_size / tick_size)
    pip_size = 10 * point if symbol_info.digits in (3, 5) else point
    value = tick_value * (pip_size / tick_size)
    return value


def calc_lot_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    symbol_info,
) -> float:
    """Calculate position size in lots.

    Formula: lots = (balance * risk_pct/100) / (SL_distance_in_pips * pip_value_per_lot)
    """
    pip_size = 10 * symbol_info.point if symbol_info.digits in (3, 5) else symbol_info.point
    sl_distance_pips = abs(entry_price - sl_price) / pip_size

    if sl_distance_pips == 0:
        logger.warning("SL distance is zero — cannot size position")
        return 0.0

    risk_money = balance * (risk_pct / 100.0)
    pv = pip_value(symbol_info)
    lots = risk_money / (sl_distance_pips * pv)

    # Round down to broker's volume step
    step = symbol_info.volume_step
    lots = math.floor(lots / step) * step

    # Clamp within broker limits
    lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

    logger.info(
        "Position size: balance=%.2f risk=%.1f%% SL_dist=%.1f pips -> %.2f lots",
        balance, risk_pct, sl_distance_pips, lots,
    )
    return round(lots, 2)


def calc_take_profit(entry: float, sl: float, rr_ratio: float, is_buy: bool) -> float:
    """Return TP price for a given Risk:Reward ratio."""
    distance = abs(entry - sl)
    if is_buy:
        return entry + distance * rr_ratio
    return entry - distance * rr_ratio
