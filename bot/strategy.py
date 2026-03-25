"""Core strategy orchestration — multi-timeframe SMC logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from bot.config import Config
from bot.mt5_connector import get_rates, get_balance, get_symbol_info, send_order
from bot.structure import (
    Trend,
    detect_swings,
    determine_trend,
    detect_market_shift_bull,
    detect_market_shift_bear,
)
from bot.zones import (
    Zone,
    find_demand_zones,
    find_supply_zones,
    nearest_supply_zone_above,
)
from bot.risk import calc_lot_size, calc_take_profit

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    direction: str          # "buy" or "sell"
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    demand_zone: Zone
    htf_trend: Trend


class SMCStrategy:
    """Smart Money Concepts multi-timeframe strategy.

    Workflow per symbol:
    1. HTF (4H) — determine trend via swing structure.
    2. MTF (1H) — locate demand/supply zones (order blocks).
    3. LTF (15m) — wait for price to enter the zone, detect liquidity
       sweep and market shift (CHoCH) for entry confirmation.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Track active zones per symbol to avoid re-entering the same zone
        self._active_zones: dict[str, list[Zone]] = {}

    # ------------------------------------------------------------------
    # Public entry point — called on every tick / timer
    # ------------------------------------------------------------------

    def evaluate(self, symbol: str) -> Optional[TradeSignal]:
        """Run the full MTF analysis for *symbol* and return a signal or None."""
        try:
            return self._evaluate(symbol)
        except Exception:
            logger.exception("Error evaluating %s", symbol)
            return None

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _evaluate(self, symbol: str) -> Optional[TradeSignal]:
        cfg = self.cfg

        # ---- Step 1: HTF trend ------------------------------------------
        htf_df = get_rates(symbol, cfg.htf_tf, count=500)
        htf_swings = detect_swings(htf_df, period=cfg.swing_period)
        trend = determine_trend(htf_swings)

        if trend == Trend.RANGE:
            logger.debug("%s HTF trend is RANGE — skipping", symbol)
            return None

        is_buy = trend == Trend.UP
        logger.info("%s HTF trend: %s", symbol, trend.value)

        # ---- Step 2: MTF zones ------------------------------------------
        mtf_df = get_rates(symbol, cfg.mtf_tf, count=500)
        mtf_swings = detect_swings(mtf_df, period=cfg.swing_period)
        mtf_highs = [s for s in mtf_swings if s.is_high]
        mtf_lows = [s for s in mtf_swings if not s.is_high]

        if is_buy:
            zones = find_demand_zones(mtf_df, mtf_highs, mtf_lows)
            zone_label = "demand"
        else:
            zones = find_supply_zones(mtf_df, mtf_highs, mtf_lows)
            zone_label = "supply"

        if not zones:
            logger.debug("%s no %s zones on MTF", symbol, zone_label)
            return None

        # Use the most recent zone
        active_zone = zones[-1]
        logger.info(
            "%s active %s zone: %.5f – %.5f (%s)",
            symbol, zone_label, active_zone.low, active_zone.high, active_zone.time,
        )

        # ---- Step 3: Is price inside the zone? --------------------------
        ltf_df = get_rates(symbol, cfg.ltf_tf, count=300)
        current_price = ltf_df["close"].iloc[-1]

        if is_buy and not active_zone.contains(current_price):
            logger.debug(
                "%s price %.5f NOT inside demand zone %.5f–%.5f",
                symbol, current_price, active_zone.low, active_zone.high,
            )
            return None
        if not is_buy and not active_zone.contains(current_price):
            logger.debug(
                "%s price %.5f NOT inside supply zone %.5f–%.5f",
                symbol, current_price, active_zone.low, active_zone.high,
            )
            return None

        # Check for liquidity sweep: price should have pierced beyond the
        # zone boundary at some point in recent LTF bars.
        recent = ltf_df.tail(20)
        if is_buy:
            swept = recent["low"].min() < active_zone.low
        else:
            swept = recent["high"].max() > active_zone.high

        if not swept:
            logger.debug("%s no liquidity sweep detected yet", symbol)
            # Still allow entry without sweep (conservative enough via CHoCH)

        # ---- Step 4: LTF Market Shift (CHoCH) confirmation -------------
        ltf_swings = detect_swings(ltf_df, period=3)  # finer period on LTF

        if is_buy:
            shift = detect_market_shift_bull(ltf_swings)
        else:
            shift = detect_market_shift_bear(ltf_swings)

        if shift is None:
            logger.debug("%s no market shift on LTF yet", symbol)
            return None

        logger.info(
            "%s market shift confirmed at %.5f (%s)",
            symbol, shift.price, shift.time,
        )

        # ---- Step 5: Build the trade ------------------------------------
        sym_info = get_symbol_info(symbol)
        point = sym_info.point
        buffer = cfg.sl_buffer_points * point

        if is_buy:
            entry = current_price
            sl = active_zone.low - buffer
            # TP: nearest supply zone or fixed RR
            supply_zones = find_supply_zones(
                mtf_df, mtf_highs, mtf_lows,
            )
            nearest_supply = nearest_supply_zone_above(entry, supply_zones)
            tp_rr = calc_take_profit(entry, sl, cfg.reward_ratio, is_buy=True)
            tp = min(nearest_supply.low, tp_rr) if nearest_supply else tp_rr
        else:
            entry = current_price
            sl = active_zone.high + buffer
            demand_zones = find_demand_zones(
                mtf_df, mtf_highs, mtf_lows,
            )
            nearest_demand = None
            demand_below = [z for z in demand_zones if z.high < entry and not z.mitigated]
            if demand_below:
                nearest_demand = max(demand_below, key=lambda z: z.high)
            tp_rr = calc_take_profit(entry, sl, cfg.reward_ratio, is_buy=False)
            tp = max(nearest_demand.high, tp_rr) if nearest_demand else tp_rr

        balance = get_balance()
        lot_size = calc_lot_size(balance, cfg.risk_percent, entry, sl, sym_info)

        if lot_size <= 0:
            logger.warning("%s lot size is 0 — trade skipped", symbol)
            return None

        # Verify RR >= required minimum
        potential_reward = abs(tp - entry)
        potential_risk = abs(entry - sl)
        actual_rr = potential_reward / potential_risk if potential_risk else 0
        if actual_rr < cfg.reward_ratio:
            logger.info(
                "%s RR %.2f below minimum %.1f — skipping",
                symbol, actual_rr, cfg.reward_ratio,
            )
            return None

        return TradeSignal(
            symbol=symbol,
            direction="buy" if is_buy else "sell",
            entry_price=round(entry, sym_info.digits),
            sl_price=round(sl, sym_info.digits),
            tp_price=round(tp, sym_info.digits),
            lot_size=lot_size,
            demand_zone=active_zone,
            htf_trend=trend,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, signal: TradeSignal) -> bool:
        """Place the order described by *signal*."""
        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "buy" else mt5.ORDER_TYPE_SELL
        result = send_order(
            symbol=signal.symbol,
            order_type=order_type,
            volume=signal.lot_size,
            price=signal.entry_price,
            sl=signal.sl_price,
            tp=signal.tp_price,
            magic=self.cfg.magic_number,
            comment=f"SMC_{signal.htf_trend.value}",
        )
        return result is not None
