#!/usr/bin/env python3
"""Entry point for the SMC Forex trading bot."""

import logging
import sys

from bot.config import Config
from bot.engine import run


def setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log"),
        ],
    )


def main() -> None:
    setup_logging()
    cfg = Config()
    logging.info(
        "Config — symbols=%s  HTF=%s  MTF=%s  LTF=%s  risk=%.1f%%  RR=1:%.1f",
        cfg.symbols, cfg.htf, cfg.mtf, cfg.ltf,
        cfg.risk_percent, cfg.reward_ratio,
    )
    run(cfg)


if __name__ == "__main__":
    main()
