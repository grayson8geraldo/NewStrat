"""Visual reporting for backtest results — equity curve and trade log."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backtest.simulator import BacktestResult

logger = logging.getLogger(__name__)


def save_trade_log(result: BacktestResult, path: str = "trade_log.csv") -> None:
    """Save all trades to a CSV file for review."""
    rows = []
    for t in result.trades:
        rows.append({
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "sl": t.sl_price,
            "tp": t.tp_price,
            "lots": t.lot_size,
            "pnl": t.pnl,
            "exit_reason": t.exit_reason,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Trade log saved to %s", path)


def plot_equity_curve(result: BacktestResult, path: str = "equity_curve.png") -> None:
    """Generate and save an equity curve chart (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed — skipping equity curve plot")
        return

    equity = [result.initial_balance]
    times = [result.trades[0].entry_time if result.trades else pd.Timestamp.now()]

    for t in result.trades:
        equity.append(equity[-1] + t.pnl)
        times.append(t.exit_time or t.entry_time)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, equity, linewidth=1.5, color="#2196F3")
    ax.fill_between(times, result.initial_balance, equity, alpha=0.1, color="#2196F3")
    ax.axhline(result.initial_balance, color="grey", linestyle="--", linewidth=0.8)

    # Mark wins and losses
    for t in result.trades:
        color = "#4CAF50" if t.pnl > 0 else "#F44336"
        marker = "^" if t.direction == "buy" else "v"
        ax.plot(t.entry_time, equity[0], marker=marker, color=color, markersize=4, alpha=0.6)

    ax.set_title("Equity Curve — SMC Strategy Backtest", fontsize=14)
    ax.set_ylabel("Balance ($)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved to %s", path)
