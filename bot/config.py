"""Configuration module — loads settings from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# MetaTrader 5 timeframe mapping (resolved at runtime via MT5 constants)
TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769,
}


@dataclass
class Config:
    # MT5 credentials
    mt5_login: int = int(os.getenv("MT5_LOGIN", "0"))
    mt5_password: str = os.getenv("MT5_PASSWORD", "")
    mt5_server: str = os.getenv("MT5_SERVER", "")

    # Symbols to trade
    symbols: list[str] = field(
        default_factory=lambda: os.getenv("SYMBOLS", "EURUSD").split(",")
    )

    # Risk management
    risk_percent: float = float(os.getenv("RISK_PERCENT", "1.0"))
    reward_ratio: float = float(os.getenv("REWARD_RATIO", "2.0"))
    magic_number: int = int(os.getenv("MAGIC_NUMBER", "123456"))

    # Timeframes (string keys into TIMEFRAME_MAP)
    htf: str = os.getenv("HTF", "H4")
    mtf: str = os.getenv("MTF", "H1")
    ltf: str = os.getenv("LTF", "M15")

    # Swing detection: number of bars on each side to confirm a swing point
    swing_period: int = int(os.getenv("SWING_PERIOD", "5"))

    # Safety buffer below demand zone low for stop-loss (in points)
    sl_buffer_points: int = int(os.getenv("SL_BUFFER_POINTS", "50"))

    @property
    def htf_tf(self) -> int:
        return TIMEFRAME_MAP[self.htf]

    @property
    def mtf_tf(self) -> int:
        return TIMEFRAME_MAP[self.mtf]

    @property
    def ltf_tf(self) -> int:
        return TIMEFRAME_MAP[self.ltf]
