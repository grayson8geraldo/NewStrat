#!/usr/bin/env python3
"""
Live paper trading with real market data — no broker account needed.

Uses Yahoo Finance for real-time prices. Trades are executed against a
virtual balance with full position tracking, SL/TP, and statistics.

Usage:
    # Forex only (default)
    python run_paper.py

    # Forex + S&P 500 futures
    python run_paper.py --symbols EURUSD,GBPUSD,ES

    # Custom balance and risk
    python run_paper.py --symbols EURUSD,ES --balance 50000 --risk 0.5

    # Resume previous session (auto if paper_state.json exists)
    python run_paper.py --symbols EURUSD

    # Reset and start fresh
    python run_paper.py --reset --symbols EURUSD,ES

Supported instruments:
    Forex:   EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD,
             EURGBP, EURJPY, GBPJPY, AUDJPY, CHFJPY
    Futures: ES (S&P 500), NQ (Nasdaq 100), YM (Dow), GC (Gold), CL (Oil)
"""

import argparse
import logging
import sys

from bot.config import Config
from live.engine import run_live_paper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMC Strategy — Live Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbols", default="EURUSD,GBPUSD",
        help="Comma-separated list of instruments (default: EURUSD,GBPUSD)",
    )
    parser.add_argument("--balance", type=float, default=200, help="Initial virtual balance (default: 200)")
    parser.add_argument("--risk", type=float, default=3.0, help="Risk %% per trade (default: 3.0)")
    parser.add_argument("--rr", type=float, default=2.0, help="Min reward:risk ratio (default: 2.0)")
    parser.add_argument("--htf", default="H4", help="Higher timeframe (default: H4)")
    parser.add_argument("--mtf", default="H1", help="Medium timeframe (default: H1)")
    parser.add_argument("--ltf", default="M15", help="Lower timeframe (default: M15)")
    parser.add_argument("--swing-period", type=int, default=5, help="Swing detection period (default: 5)")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between evaluations (default: 60)")
    parser.add_argument("--reset", action="store_true", help="Reset paper broker state and start fresh")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("paper_trading.log"),
        ],
    )

    # Config
    cfg = Config()
    cfg.risk_percent = args.risk
    cfg.reward_ratio = args.rr
    cfg.htf = args.htf
    cfg.mtf = args.mtf
    cfg.ltf = args.ltf
    cfg.swing_period = args.swing_period

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Reset if requested
    if args.reset:
        from live.paper_broker import PaperBroker
        broker = PaperBroker(initial_balance=args.balance)
        broker.reset()
        print("Paper broker state reset.\n")

    run_live_paper(
        cfg=cfg,
        symbols=symbols,
        initial_balance=args.balance,
        poll_interval=args.poll,
    )


if __name__ == "__main__":
    main()
