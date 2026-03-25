"""Main event loop — runs the strategy on a timer."""

from __future__ import annotations

import logging
import time

import MetaTrader5 as mt5

from bot.config import Config
from bot.mt5_connector import connect, disconnect
from bot.strategy import SMCStrategy

logger = logging.getLogger(__name__)

# Minimum seconds between evaluations for the same symbol
COOLDOWN_SECONDS = 60


def has_open_position(symbol: str, magic: int) -> bool:
    """Check if there is already an open position for *symbol* with our magic."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == magic for p in positions)


def run(cfg: Config) -> None:
    """Connect to MT5 and enter the main loop."""
    if not connect(cfg):
        raise SystemExit("Failed to connect to MT5")

    strategy = SMCStrategy(cfg)
    last_eval: dict[str, float] = {}

    logger.info("Bot started — monitoring %s", cfg.symbols)

    try:
        while True:
            now = time.time()

            for symbol in cfg.symbols:
                # Cooldown check
                if now - last_eval.get(symbol, 0) < COOLDOWN_SECONDS:
                    continue
                last_eval[symbol] = now

                # Skip if already in a trade
                if has_open_position(symbol, cfg.magic_number):
                    logger.debug("%s already has an open position", symbol)
                    continue

                signal = strategy.evaluate(symbol)
                if signal is None:
                    continue

                logger.info(
                    "SIGNAL %s %s @ %.5f  SL=%.5f  TP=%.5f  lots=%.2f",
                    signal.direction.upper(),
                    signal.symbol,
                    signal.entry_price,
                    signal.sl_price,
                    signal.tp_price,
                    signal.lot_size,
                )

                success = strategy.execute(signal)
                if success:
                    logger.info("Trade placed for %s", symbol)
                else:
                    logger.error("Failed to place trade for %s", symbol)

            # Sleep until next evaluation cycle
            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        disconnect()
