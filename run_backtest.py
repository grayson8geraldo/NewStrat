#!/usr/bin/env python3
"""
Run a backtest of the SMC strategy on historical Forex data.

Usage:
    python run_backtest.py                                    # synthetic data (no internet needed)
    python run_backtest.py --symbol GBPUSD --seed 123         # different pair / seed
    python run_backtest.py --yahoo --symbol EURUSD --period 60d  # real data from Yahoo Finance
    python run_backtest.py --csv data/EURUSD_M15.csv          # your own CSV file

Data sources (in order of priority):
  1. --csv <path>       : Load from a local CSV file
  2. --yahoo            : Download from Yahoo Finance (requires internet)
  3. (default)          : Generate realistic synthetic data (works offline)
"""

import argparse
import logging
import sys

from bot.config import Config
from backtest.simulator import run_backtest
from backtest.report import save_trade_log, plot_equity_curve


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=[logging.StreamHandler(sys.stdout)])


def main() -> None:
    parser = argparse.ArgumentParser(description="SMC Strategy Backtester")
    parser.add_argument("--symbol", default="EURUSD", help="Forex pair (default: EURUSD)")
    parser.add_argument("--balance", type=float, default=200, help="Initial balance (default: 200)")
    parser.add_argument("--risk", type=float, default=3.0, help="Risk percent per trade (default: 3.0)")
    parser.add_argument("--rr", type=float, default=2.0, help="Reward:Risk ratio (default: 2.0)")
    parser.add_argument("--htf", default="H4", help="Higher timeframe (default: H4)")
    parser.add_argument("--mtf", default="H1", help="Medium timeframe (default: H1)")
    parser.add_argument("--ltf", default="M15", help="Lower timeframe (default: M15)")
    parser.add_argument("--swing-period", type=int, default=5, help="Swing detection period (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # Data source options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--yahoo", action="store_true",
                            help="Download real data from Yahoo Finance")
    data_group.add_argument("--csv", type=str, default=None,
                            help="Path to a CSV file with OHLCV data (15m bars)")
    parser.add_argument("--period", default="60d",
                        help="Yahoo Finance period: 60d, 6mo, 1y (only with --yahoo)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic data generation (default: 42)")
    parser.add_argument("--bars", type=int, default=4000,
                        help="Number of LTF bars for synthetic data (default: 4000)")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Build config from CLI args
    cfg = Config()
    cfg.risk_percent = args.risk
    cfg.reward_ratio = args.rr
    cfg.htf = args.htf
    cfg.mtf = args.mtf
    cfg.ltf = args.ltf
    cfg.swing_period = args.swing_period

    # ---- Load data -------------------------------------------------------
    if args.csv:
        from backtest.data_provider import load_csv
        print(f"\nLoading data from {args.csv}…\n")
        ltf_df = load_csv(args.csv)
        # Resample to MTF and HTF
        ltf_indexed = ltf_df.set_index("time")
        mtf_df = ltf_indexed.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna().reset_index()
        htf_df = ltf_indexed.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna().reset_index()

    elif args.yahoo:
        from backtest.data_provider import download_yahoo
        print(f"\nDownloading data for {args.symbol} from Yahoo Finance…\n")
        try:
            htf_df = download_yahoo(args.symbol, cfg.htf, period=args.period)
            mtf_df = download_yahoo(args.symbol, cfg.mtf, period=args.period)
            ltf_df = download_yahoo(args.symbol, cfg.ltf, period=args.period)
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("\nFalling back to synthetic data…\n")
            from backtest.synthetic_data import generate_mtf_data
            htf_df, mtf_df, ltf_df = generate_mtf_data(args.symbol, seed=args.seed)

    else:
        # Default: synthetic data
        from backtest.synthetic_data import generate_mtf_data, generate_forex_data
        print(f"\nGenerating synthetic data for {args.symbol} (seed={args.seed}, {args.bars} LTF bars)…\n")

        ltf_df = generate_forex_data(
            symbol=args.symbol,
            timeframe_minutes=15,
            num_bars=args.bars,
            seed=args.seed,
        )
        ltf_indexed = ltf_df.set_index("time")
        mtf_df = ltf_indexed.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna().reset_index()
        htf_df = ltf_indexed.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna().reset_index()

    print(f"  HTF ({cfg.htf}): {len(htf_df)} bars")
    print(f"  MTF ({cfg.mtf}): {len(mtf_df)} bars")
    print(f"  LTF ({cfg.ltf}): {len(ltf_df)} bars")
    print()

    # ---- Run backtest ----------------------------------------------------
    result = run_backtest(
        htf_df=htf_df,
        mtf_df=mtf_df,
        ltf_df=ltf_df,
        cfg=cfg,
        symbol=args.symbol,
        initial_balance=args.balance,
    )

    # ---- Output results --------------------------------------------------
    print()
    print(result.summary())

    if result.trades:
        save_trade_log(result)
        plot_equity_curve(result)
        print("\nFiles saved: trade_log.csv, equity_curve.png")
    else:
        print("\nNo trades were generated during this period.")
        print("Try: --seed <N> to change data, or --swing-period 3 for more signals.")


if __name__ == "__main__":
    main()
