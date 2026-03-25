"""Trade simulator — walks through historical bars and applies the SMC strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from bot.config import Config
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
from bot.risk import calc_take_profit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated symbol info (mirrors MT5 symbol_info for pip/lot calculations)
# ---------------------------------------------------------------------------

@dataclass
class SimSymbolInfo:
    """Minimal replica of mt5.SymbolInfo for the risk module."""
    symbol: str = "EURUSD"
    point: float = 0.00001
    digits: int = 5
    trade_tick_size: float = 0.00001
    trade_tick_value: float = 1.0   # $1 per tick on a standard lot
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0

    @classmethod
    def for_symbol(cls, symbol: str) -> "SimSymbolInfo":
        """Return sensible defaults for well-known Forex pairs."""
        if "JPY" in symbol:
            return cls(
                symbol=symbol,
                point=0.001,
                digits=3,
                trade_tick_size=0.001,
                trade_tick_value=1000.0 / 100,  # ≈ $10 per pip for standard lot
            )
        return cls(symbol=symbol)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    symbol: str
    direction: str          # "buy" or "sell"
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    entry_time: pd.Timestamp
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    pnl: float = 0.0       # in deposit currency
    exit_reason: str = ""   # "tp", "sl", "end"


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    trades: list[SimTrade] = field(default_factory=list)
    initial_balance: float = 10_000.0
    final_balance: float = 10_000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losers(self) -> int:
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades * 100 if self.total_trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = self.initial_balance
        peak = equity
        max_dd = 0.0
        for t in self.trades:
            equity += t.pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "           BACKTEST RESULTS",
            "=" * 55,
            f"  Initial balance:   ${self.initial_balance:,.2f}",
            f"  Final balance:     ${self.final_balance:,.2f}",
            f"  Net P&L:           ${self.total_pnl:,.2f}",
            f"  Return:            {self.total_pnl / self.initial_balance * 100:+.2f}%",
            "-" * 55,
            f"  Total trades:      {self.total_trades}",
            f"  Winners:           {self.winners}",
            f"  Losers:            {self.losers}",
            f"  Win rate:          {self.win_rate:.1f}%",
            f"  Profit factor:     {self.profit_factor:.2f}",
            f"  Max drawdown:      {self.max_drawdown:.2f}%",
            "=" * 55,
        ]
        return "\n".join(lines)


def _calc_sim_lot_size(
    balance: float,
    risk_pct: float,
    entry: float,
    sl: float,
    sym_info: SimSymbolInfo,
) -> float:
    """Position sizing for the simulator (mirrors bot.risk.calc_lot_size)."""
    from bot.risk import calc_lot_size
    return calc_lot_size(balance, risk_pct, entry, sl, sym_info)


def run_backtest(
    htf_df: pd.DataFrame,
    mtf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    cfg: Config,
    symbol: str = "EURUSD",
    initial_balance: float = 10_000.0,
) -> BacktestResult:
    """Walk forward through LTF bars, applying the full SMC strategy.

    The HTF and MTF are analysed using all data up to the current LTF bar's
    timestamp (look-ahead bias free).
    """
    sym_info = SimSymbolInfo.for_symbol(symbol)
    balance = initial_balance
    trades: list[SimTrade] = []
    open_trade: SimTrade | None = None

    # Pre-sort
    htf_df = htf_df.sort_values("time").reset_index(drop=True)
    mtf_df = mtf_df.sort_values("time").reset_index(drop=True)
    ltf_df = ltf_df.sort_values("time").reset_index(drop=True)

    # We need a minimum warm-up period
    min_ltf_bars = 50
    if len(ltf_df) < min_ltf_bars:
        logger.warning("Not enough LTF data (%d bars) for backtest", len(ltf_df))
        return BacktestResult(trades=[], initial_balance=initial_balance, final_balance=balance)

    logger.info(
        "Starting backtest: %s | %d HTF bars, %d MTF bars, %d LTF bars | balance=$%.2f",
        symbol, len(htf_df), len(mtf_df), len(ltf_df), balance,
    )

    for i in range(min_ltf_bars, len(ltf_df)):
        bar = ltf_df.iloc[i]
        bar_time = bar["time"]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        # ---- Check open trade for SL / TP hit ----------------------------
        if open_trade is not None:
            hit_sl = False
            hit_tp = False

            if open_trade.direction == "buy":
                if bar_low <= open_trade.sl_price:
                    hit_sl = True
                if bar_high >= open_trade.tp_price:
                    hit_tp = True
            else:
                if bar_high >= open_trade.sl_price:
                    hit_sl = True
                if bar_low <= open_trade.tp_price:
                    hit_tp = True

            if hit_sl and hit_tp:
                # Assume SL hit first (conservative)
                hit_tp = False

            if hit_sl:
                open_trade.exit_price = open_trade.sl_price
                open_trade.exit_time = bar_time
                open_trade.exit_reason = "sl"
            elif hit_tp:
                open_trade.exit_price = open_trade.tp_price
                open_trade.exit_time = bar_time
                open_trade.exit_reason = "tp"

            if open_trade.exit_price is not None:
                # Calculate P&L
                pip_size = 10 * sym_info.point if sym_info.digits in (3, 5) else sym_info.point
                if open_trade.direction == "buy":
                    pips = (open_trade.exit_price - open_trade.entry_price) / pip_size
                else:
                    pips = (open_trade.entry_price - open_trade.exit_price) / pip_size

                pnl = pips * open_trade.lot_size * (sym_info.trade_tick_value * pip_size / sym_info.trade_tick_size)
                open_trade.pnl = round(pnl, 2)
                balance += open_trade.pnl
                trades.append(open_trade)
                logger.info(
                    "  [%s] %s %s closed @ %.5f (%s) P&L=$%.2f  Balance=$%.2f",
                    bar_time, open_trade.direction.upper(), symbol,
                    open_trade.exit_price, open_trade.exit_reason,
                    open_trade.pnl, balance,
                )
                open_trade = None

            # If still in a trade, skip signal search
            if open_trade is not None:
                continue

        # ---- Step 1: HTF trend (only bars up to current time) -------------
        htf_visible = htf_df[htf_df["time"] <= bar_time]
        if len(htf_visible) < 20:
            continue

        htf_swings = detect_swings(htf_visible, period=cfg.swing_period)
        trend = determine_trend(htf_swings)
        if trend == Trend.RANGE:
            continue

        is_buy = trend == Trend.UP

        # ---- Step 2: MTF zones --------------------------------------------
        mtf_visible = mtf_df[mtf_df["time"] <= bar_time]
        if len(mtf_visible) < 30:
            continue

        mtf_swings = detect_swings(mtf_visible, period=cfg.swing_period)
        mtf_highs = [s for s in mtf_swings if s.is_high]
        mtf_lows = [s for s in mtf_swings if not s.is_high]

        if is_buy:
            zones = find_demand_zones(mtf_visible, mtf_highs, mtf_lows)
        else:
            zones = find_supply_zones(mtf_visible, mtf_highs, mtf_lows)

        if not zones:
            continue

        active_zone = zones[-1]

        # ---- Step 3: Price inside the zone? --------------------------------
        if is_buy and not active_zone.contains(bar_close):
            continue
        if not is_buy and not active_zone.contains(bar_close):
            continue

        # ---- Step 4: LTF Market Shift (CHoCH) ------------------------------
        ltf_window = ltf_df.iloc[max(0, i - 40): i + 1]
        ltf_swings = detect_swings(ltf_window, period=3)

        if is_buy:
            shift = detect_market_shift_bull(ltf_swings)
        else:
            shift = detect_market_shift_bear(ltf_swings)

        if shift is None:
            continue

        # ---- Step 5: Open simulated trade ----------------------------------
        entry = bar_close
        buffer = cfg.sl_buffer_points * sym_info.point

        if is_buy:
            sl = active_zone.low - buffer
            # TP: nearest supply zone or fixed RR
            supply_zones = find_supply_zones(mtf_visible, mtf_highs, mtf_lows)
            nearest_supply = nearest_supply_zone_above(entry, supply_zones)
            tp_rr = calc_take_profit(entry, sl, cfg.reward_ratio, is_buy=True)
            tp = min(nearest_supply.low, tp_rr) if nearest_supply else tp_rr
        else:
            sl = active_zone.high + buffer
            demand_zones = find_demand_zones(mtf_visible, mtf_highs, mtf_lows)
            demand_below = [z for z in demand_zones if z.high < entry and not z.mitigated]
            nearest_demand = max(demand_below, key=lambda z: z.high) if demand_below else None
            tp_rr = calc_take_profit(entry, sl, cfg.reward_ratio, is_buy=False)
            tp = max(nearest_demand.high, tp_rr) if nearest_demand else tp_rr

        # Check RR
        potential_reward = abs(tp - entry)
        potential_risk = abs(entry - sl)
        actual_rr = potential_reward / potential_risk if potential_risk else 0
        if actual_rr < cfg.reward_ratio:
            continue

        lot_size = _calc_sim_lot_size(balance, cfg.risk_percent, entry, sl, sym_info)
        if lot_size <= 0:
            continue

        open_trade = SimTrade(
            symbol=symbol,
            direction="buy" if is_buy else "sell",
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            lot_size=lot_size,
            entry_time=bar_time,
        )
        logger.info(
            "  [%s] OPEN %s %s @ %.5f  SL=%.5f  TP=%.5f  lots=%.2f",
            bar_time, open_trade.direction.upper(), symbol,
            entry, sl, tp, lot_size,
        )

    # Close any trade still open at the end
    if open_trade is not None:
        last_bar = ltf_df.iloc[-1]
        open_trade.exit_price = last_bar["close"]
        open_trade.exit_time = last_bar["time"]
        open_trade.exit_reason = "end"
        pip_size = 10 * sym_info.point if sym_info.digits in (3, 5) else sym_info.point
        if open_trade.direction == "buy":
            pips = (open_trade.exit_price - open_trade.entry_price) / pip_size
        else:
            pips = (open_trade.entry_price - open_trade.exit_price) / pip_size
        pnl = pips * open_trade.lot_size * (sym_info.trade_tick_value * pip_size / sym_info.trade_tick_size)
        open_trade.pnl = round(pnl, 2)
        balance += open_trade.pnl
        trades.append(open_trade)

    result = BacktestResult(
        trades=trades,
        initial_balance=initial_balance,
        final_balance=round(balance, 2),
    )
    return result
