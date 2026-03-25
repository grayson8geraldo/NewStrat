"""Live paper trading engine — polls real market data and executes the SMC strategy."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from bot.config import Config
from bot.structure import (
    Trend,
    detect_swings,
    determine_trend,
    detect_market_shift_bull,
    detect_market_shift_bear,
)
from bot.zones import (
    find_demand_zones,
    find_supply_zones,
    nearest_supply_zone_above,
)
from bot.risk import calc_take_profit
from live.feed import fetch_bars, get_meta
from live.paper_broker import PaperBroker

logger = logging.getLogger(__name__)


def _print_status(broker: PaperBroker, symbols: list[str]) -> None:
    """Print a compact status line."""
    stats = broker.stats()
    now = datetime.utcnow().strftime("%H:%M:%S UTC")
    open_str = ""
    for p in broker.positions:
        arrow = "+" if p.unrealized_pnl >= 0 else ""
        open_str += f"  #{p.id} {p.direction.upper()} {p.symbol} {arrow}${p.unrealized_pnl:.2f}"

    print(
        f"\r[{now}]  Balance: ${stats['balance']:,.2f}  "
        f"Equity: ${stats['equity']:,.2f}  "
        f"Trades: {stats['total_trades']}  "
        f"W/L: {stats['winners']}/{stats['losers']}  "
        f"P&L: ${stats['net_pnl']:+,.2f} ({stats['return_pct']:+.2f}%)"
        f"{open_str}",
        end="    \n",
    )


def run_live_paper(
    cfg: Config,
    symbols: list[str],
    initial_balance: float = 10_000.0,
    poll_interval: int = 60,
) -> None:
    """Main loop: fetch real data, analyse, trade on paper.

    Args:
        cfg: Strategy configuration
        symbols: List of instruments to trade (e.g. ["EURUSD", "ES"])
        initial_balance: Starting virtual balance
        poll_interval: Seconds between data polls (default: 60)
    """
    broker = PaperBroker(initial_balance=initial_balance)

    print("=" * 60)
    print("  SMC Strategy — LIVE PAPER TRADING")
    print("=" * 60)
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Timeframes: HTF={cfg.htf}  MTF={cfg.mtf}  LTF={cfg.ltf}")
    print(f"  Risk:       {cfg.risk_percent}% per trade   R:R = 1:{cfg.reward_ratio}")
    print(f"  Balance:    ${initial_balance:,.2f} (virtual)")
    print(f"  Risk/trade: ${initial_balance * cfg.risk_percent / 100:,.2f}")
    print(f"  Poll:       every {poll_interval}s")

    if initial_balance < 500 and cfg.risk_percent < 2:
        print()
        print("  WARNING: Balance < $500 with risk < 2%.")
        print("  Minimum lot sizes may prevent trade execution.")
        print("  Consider --risk 3 or higher for small accounts.")

    if cfg.risk_percent > 5:
        print()
        print("  WARNING: Risk > 5% per trade is very aggressive.")
        print("  A string of losses can wipe out the account quickly.")

    print("=" * 60)
    print("  Press Ctrl+C to stop\n")

    cycle = 0
    try:
        while True:
            cycle += 1

            for symbol in symbols:
                try:
                    _process_symbol(cfg, broker, symbol)
                except Exception as e:
                    logger.error("Error processing %s: %s", symbol, e)

            # Update unrealized P&L
            prices = {}
            for symbol in symbols:
                try:
                    ltf = fetch_bars(symbol, cfg.ltf, count=5)
                    prices[symbol] = ltf["close"].iloc[-1]
                except Exception:
                    pass
            broker.update_unrealized(prices)

            # Print status every cycle
            _print_status(broker, symbols)

            # Detailed stats every 10 cycles
            if cycle % 10 == 0:
                _print_full_stats(broker)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading…")
        _print_full_stats(broker)
        print(f"\nState saved to {broker._state_file} — run again to resume.")


def _process_symbol(cfg: Config, broker: PaperBroker, symbol: str) -> None:
    """Run one evaluation cycle for a single symbol."""
    meta = get_meta(symbol)

    # ---- Fetch data for all 3 timeframes ---------------------------------
    htf_df = fetch_bars(symbol, cfg.htf, count=500)
    mtf_df = fetch_bars(symbol, cfg.mtf, count=500)
    ltf_df = fetch_bars(symbol, cfg.ltf, count=300)

    current_bar = ltf_df.iloc[-1]
    current_price = current_bar["close"]
    current_high = current_bar["high"]
    current_low = current_bar["low"]

    # ---- Check SL/TP on open positions ------------------------------------
    broker.check_sl_tp(symbol, current_high, current_low)

    # ---- Skip if already in a position for this symbol --------------------
    if broker.has_position(symbol):
        return

    # ---- Step 1: HTF trend ------------------------------------------------
    htf_swings = detect_swings(htf_df, period=cfg.swing_period)
    trend = determine_trend(htf_swings)

    if trend == Trend.RANGE:
        logger.debug("%s HTF: RANGE — skip", symbol)
        return

    is_buy = trend == Trend.UP
    logger.info("%s HTF trend: %s", symbol, trend.value)

    # ---- Step 2: MTF zones ------------------------------------------------
    mtf_swings = detect_swings(mtf_df, period=cfg.swing_period)
    mtf_highs = [s for s in mtf_swings if s.is_high]
    mtf_lows = [s for s in mtf_swings if not s.is_high]

    if is_buy:
        zones = find_demand_zones(mtf_df, mtf_highs, mtf_lows)
    else:
        zones = find_supply_zones(mtf_df, mtf_highs, mtf_lows)

    if not zones:
        logger.debug("%s no zones on MTF", symbol)
        return

    active_zone = zones[-1]

    # ---- Step 3: Price inside the zone? -----------------------------------
    if not active_zone.contains(current_price):
        logger.debug(
            "%s price %.5f outside zone %.5f–%.5f",
            symbol, current_price, active_zone.low, active_zone.high,
        )
        return

    logger.info(
        "%s price %.5f INSIDE %s zone %.5f–%.5f",
        symbol, current_price, active_zone.zone_type, active_zone.low, active_zone.high,
    )

    # ---- Step 4: LTF Market Shift (CHoCH) ---------------------------------
    ltf_swings = detect_swings(ltf_df, period=3)

    if is_buy:
        shift = detect_market_shift_bull(ltf_swings)
    else:
        shift = detect_market_shift_bear(ltf_swings)

    if shift is None:
        logger.debug("%s no market shift on LTF", symbol)
        return

    logger.info("%s MARKET SHIFT confirmed at %.5f", symbol, shift.price)

    # ---- Step 5: Calculate trade parameters --------------------------------
    buffer = cfg.sl_buffer_points * meta["point"]

    if is_buy:
        sl = active_zone.low - buffer
        supply_zones = find_supply_zones(mtf_df, mtf_highs, mtf_lows)
        nearest_supply = nearest_supply_zone_above(current_price, supply_zones)
        tp_rr = calc_take_profit(current_price, sl, cfg.reward_ratio, is_buy=True)
        tp = min(nearest_supply.low, tp_rr) if nearest_supply else tp_rr
    else:
        sl = active_zone.high + buffer
        demand_zones = find_demand_zones(mtf_df, mtf_highs, mtf_lows)
        demand_below = [z for z in demand_zones if z.high < current_price and not z.mitigated]
        nearest_demand = max(demand_below, key=lambda z: z.high) if demand_below else None
        tp_rr = calc_take_profit(current_price, sl, cfg.reward_ratio, is_buy=False)
        tp = max(nearest_demand.high, tp_rr) if nearest_demand else tp_rr

    # Verify R:R
    reward = abs(tp - current_price)
    risk = abs(current_price - sl)
    rr = reward / risk if risk else 0
    if rr < cfg.reward_ratio:
        logger.info("%s R:R %.2f < %.1f — skip", symbol, rr, cfg.reward_ratio)
        return

    # Position sizing
    lot_size = broker.calc_lot_size(symbol, current_price, sl, cfg.risk_percent)
    if lot_size <= 0:
        logger.warning("%s lot size = 0 — skip", symbol)
        return

    # ---- Execute paper trade ------------------------------------------------
    direction = "buy" if is_buy else "sell"
    broker.open_position(
        symbol=symbol,
        direction=direction,
        volume=lot_size,
        price=current_price,
        sl=sl,
        tp=tp,
        comment=f"SMC_{trend.value}",
    )


def _print_full_stats(broker: PaperBroker) -> None:
    """Print detailed statistics."""
    stats = broker.stats()
    print()
    print("-" * 50)
    print("  PAPER TRADING STATS")
    print("-" * 50)
    print(f"  Balance:        ${stats['balance']:,.2f}")
    print(f"  Equity:         ${stats['equity']:,.2f}")
    print(f"  Net P&L:        ${stats['net_pnl']:+,.2f} ({stats['return_pct']:+.2f}%)")
    print(f"  Total trades:   {stats['total_trades']}")
    print(f"  Win rate:       {stats['win_rate']}%")
    print(f"  Profit factor:  {stats['profit_factor']}")
    print(f"  Open positions: {stats['open_positions']}")

    if broker.positions:
        print("\n  Open positions:")
        for p in broker.positions:
            pnl_str = f"${p.unrealized_pnl:+.2f}" if p.unrealized_pnl else "…"
            print(
                f"    #{p.id} {p.direction.upper()} {p.symbol} "
                f"{p.volume} lots @ {p.entry_price}  "
                f"SL={p.sl}  TP={p.tp}  P&L={pnl_str}"
            )

    if broker.history:
        print(f"\n  Last 5 trades:")
        for t in broker.history[-5:]:
            emoji = "W" if t.pnl > 0 else "L"
            print(
                f"    [{emoji}] #{t.id} {t.direction.upper()} {t.symbol} "
                f"@ {t.entry_price} -> {t.exit_price}  "
                f"P&L=${t.pnl:+.2f} ({t.exit_reason})"
            )
    print("-" * 50)
