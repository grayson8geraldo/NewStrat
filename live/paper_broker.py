"""Paper trading broker — simulates order execution with virtual balance.

Tracks open positions, pending orders, closed trades, equity, margin, etc.
Uses real market prices from the feed module.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from live.feed import get_meta

logger = logging.getLogger(__name__)

STATE_FILE = "paper_state.json"


@dataclass
class Position:
    id: int
    symbol: str
    direction: str          # "buy" or "sell"
    volume: float           # lots or contracts
    entry_price: float
    sl: float
    tp: float
    entry_time: str
    comment: str = ""
    unrealized_pnl: float = 0.0


@dataclass
class ClosedTrade:
    id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    entry_time: str
    exit_time: str
    pnl: float
    exit_reason: str        # "tp", "sl", "manual", "signal"


class PaperBroker:
    """Virtual broker that executes trades against real market prices."""

    def __init__(self, initial_balance: float = 10_000.0, state_file: str = STATE_FILE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: list[Position] = []
        self.history: list[ClosedTrade] = []
        self._next_id = 1
        self._state_file = state_file

        # Try to resume from saved state
        self._load_state()

    # ---- Order execution -------------------------------------------------

    def open_position(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> Position:
        """Open a new paper position at *price*."""
        meta = get_meta(symbol)

        # Apply simulated spread
        spread = meta["spread_pips"] * meta["pip_size"]
        if direction == "buy":
            fill_price = price + spread / 2
        else:
            fill_price = price - spread / 2

        pos = Position(
            id=self._next_id,
            symbol=symbol,
            direction=direction,
            volume=volume,
            entry_price=round(fill_price, meta["digits"]),
            sl=round(sl, meta["digits"]),
            tp=round(tp, meta["digits"]),
            entry_time=datetime.utcnow().isoformat(),
            comment=comment,
        )
        self._next_id += 1
        self.positions.append(pos)
        self._save_state()

        logger.info(
            "PAPER OPEN #%d: %s %s %.2f @ %.5f  SL=%.5f  TP=%.5f  [%s]",
            pos.id, direction.upper(), symbol, volume,
            pos.entry_price, sl, tp, comment,
        )
        return pos

    def close_position(self, pos_id: int, price: float, reason: str = "manual") -> Optional[ClosedTrade]:
        """Close an open position at *price*."""
        pos = next((p for p in self.positions if p.id == pos_id), None)
        if pos is None:
            logger.warning("Position #%d not found", pos_id)
            return None

        meta = get_meta(pos.symbol)
        spread = meta["spread_pips"] * meta["pip_size"]

        # Apply spread on exit
        if pos.direction == "buy":
            exit_price = price - spread / 2
        else:
            exit_price = price + spread / 2

        pnl = self._calc_pnl(pos, exit_price)

        trade = ClosedTrade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            volume=pos.volume,
            entry_price=pos.entry_price,
            exit_price=round(exit_price, meta["digits"]),
            sl=pos.sl,
            tp=pos.tp,
            entry_time=pos.entry_time,
            exit_time=datetime.utcnow().isoformat(),
            pnl=round(pnl, 2),
            exit_reason=reason,
        )

        self.balance += trade.pnl
        self.positions = [p for p in self.positions if p.id != pos_id]
        self.history.append(trade)
        self._save_state()

        logger.info(
            "PAPER CLOSE #%d: %s %s @ %.5f -> %.5f  P&L=$%.2f (%s)  Balance=$%.2f",
            trade.id, trade.direction.upper(), trade.symbol,
            trade.entry_price, trade.exit_price,
            trade.pnl, reason, self.balance,
        )
        return trade

    # ---- SL / TP checking ------------------------------------------------

    def check_sl_tp(self, symbol: str, current_high: float, current_low: float) -> list[ClosedTrade]:
        """Check all open positions for *symbol* against the current bar's high/low.

        Returns list of trades that were closed.
        """
        closed = []
        positions_for_symbol = [p for p in self.positions if p.symbol == symbol]

        for pos in positions_for_symbol:
            exit_price = None
            reason = ""

            if pos.direction == "buy":
                if current_low <= pos.sl:
                    exit_price = pos.sl
                    reason = "sl"
                elif current_high >= pos.tp:
                    exit_price = pos.tp
                    reason = "tp"
            else:  # sell
                if current_high >= pos.sl:
                    exit_price = pos.sl
                    reason = "sl"
                elif current_low <= pos.tp:
                    exit_price = pos.tp
                    reason = "tp"

            if exit_price is not None:
                trade = self.close_position(pos.id, exit_price, reason)
                if trade:
                    closed.append(trade)

        return closed

    # ---- Portfolio stats -------------------------------------------------

    def update_unrealized(self, prices: dict[str, float]) -> None:
        """Update unrealized P&L on all open positions."""
        for pos in self.positions:
            if pos.symbol in prices:
                pos.unrealized_pnl = round(self._calc_pnl(pos, prices[pos.symbol]), 2)

    @property
    def equity(self) -> float:
        return self.balance + sum(p.unrealized_pnl for p in self.positions)

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions)

    def stats(self) -> dict:
        """Return summary statistics."""
        wins = [t for t in self.history if t.pnl > 0]
        losses = [t for t in self.history if t.pnl < 0]
        total = len(self.history)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        return {
            "initial_balance": self.initial_balance,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "open_positions": self.open_count,
            "total_trades": total,
            "winners": len(wins),
            "losers": len(losses),
            "win_rate": round(len(wins) / total * 100, 1) if total else 0,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0,
            "net_pnl": round(self.balance - self.initial_balance, 2),
            "return_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
        }

    # ---- Position sizing -------------------------------------------------

    def calc_lot_size(self, symbol: str, entry: float, sl: float, risk_pct: float = 1.0) -> float:
        """Calculate position size based on risk percentage."""
        meta = get_meta(symbol)
        pip_size = meta["pip_size"]
        sl_pips = abs(entry - sl) / pip_size

        if sl_pips == 0:
            return 0.0

        risk_money = self.balance * (risk_pct / 100.0)
        tick_value = meta["tick_value"]

        # For forex: tick_value is $ per pip per standard lot
        # lots = risk_money / (sl_pips * tick_value_per_pip)
        lots = risk_money / (sl_pips * tick_value)

        step = meta["volume_step"]
        lots = math.floor(lots / step) * step
        lots = max(meta["volume_min"], min(lots, meta["volume_max"]))
        return round(lots, 2)

    # ---- Internal helpers ------------------------------------------------

    def _calc_pnl(self, pos: Position, exit_price: float) -> float:
        meta = get_meta(pos.symbol)
        pip_size = meta["pip_size"]
        tick_value = meta["tick_value"]

        if pos.direction == "buy":
            pips = (exit_price - pos.entry_price) / pip_size
        else:
            pips = (pos.entry_price - exit_price) / pip_size

        return pips * pos.volume * tick_value

    # ---- State persistence -----------------------------------------------

    def _save_state(self) -> None:
        """Save broker state to JSON for crash recovery."""
        state = {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "next_id": self._next_id,
            "positions": [asdict(p) for p in self.positions],
            "history": [asdict(t) for t in self.history],
        }
        Path(self._state_file).write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        """Resume from saved state if available."""
        path = Path(self._state_file)
        if not path.exists():
            return

        try:
            state = json.loads(path.read_text())
            self.initial_balance = state["initial_balance"]
            self.balance = state["balance"]
            self._next_id = state["next_id"]
            self.positions = [Position(**p) for p in state["positions"]]
            self.history = [ClosedTrade(**t) for t in state["history"]]
            logger.info(
                "Resumed paper broker: balance=$%.2f, %d open positions, %d historical trades",
                self.balance, len(self.positions), len(self.history),
            )
        except Exception:
            logger.warning("Could not load state from %s — starting fresh", self._state_file)

    def reset(self) -> None:
        """Reset to initial state."""
        self.balance = self.initial_balance
        self.positions = []
        self.history = []
        self._next_id = 1
        self._save_state()
        logger.info("Paper broker reset — balance=$%.2f", self.balance)
