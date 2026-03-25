"""Thin wrapper around MetaTrader5 for initialisation, data fetching and order execution."""

from __future__ import annotations

import logging
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from bot.config import Config

logger = logging.getLogger(__name__)


def connect(cfg: Config) -> bool:
    """Initialise and log in to the MT5 terminal."""
    if not mt5.initialize():
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        return False

    if cfg.mt5_login:
        authorised = mt5.login(
            login=cfg.mt5_login,
            password=cfg.mt5_password,
            server=cfg.mt5_server,
        )
        if not authorised:
            logger.error("MT5 login failed: %s", mt5.last_error())
            return False

    logger.info("Connected to MT5 — account %s", mt5.account_info().login)
    return True


def disconnect() -> None:
    mt5.shutdown()


def get_rates(symbol: str, timeframe: int, count: int = 500) -> pd.DataFrame:
    """Return OHLCV DataFrame for *symbol* / *timeframe*."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol} TF={timeframe}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def get_balance() -> float:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("Cannot read account info")
    return info.balance


def get_symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info unavailable for {symbol}")
    return info


def send_order(
    symbol: str,
    order_type: int,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    magic: int,
    comment: str = "SMC_Bot",
) -> Optional[mt5.OrderSendResult]:
    """Send a market order and return the result."""
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("Order failed: %s", result)
        return None
    logger.info(
        "Order executed — %s %s %.5f lots @ %.5f  SL=%.5f  TP=%.5f",
        "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL",
        symbol, volume, price, sl, tp,
    )
    return result
