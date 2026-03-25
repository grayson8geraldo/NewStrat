"""Unit tests for the paper broker."""

import os
import json
from pathlib import Path

import pytest

# Patch feed.get_meta before importing PaperBroker
from live import feed

_MOCK_META = {
    "type": "forex",
    "point": 0.00001,
    "digits": 5,
    "pip_size": 0.0001,
    "tick_value": 10.0,
    "volume_step": 0.01,
    "volume_min": 0.01,
    "volume_max": 100.0,
    "spread_pips": 0.0,  # zero spread for predictable tests
}


@pytest.fixture(autouse=True)
def patch_meta(monkeypatch):
    monkeypatch.setattr(feed, "SYMBOL_META", {"EURUSD": _MOCK_META, "ES": _MOCK_META})


@pytest.fixture
def broker(tmp_path):
    from live.paper_broker import PaperBroker
    state_file = str(tmp_path / "test_state.json")
    return PaperBroker(initial_balance=10_000.0, state_file=state_file)


class TestOpenClose:
    def test_open_and_close_profit(self, broker):
        pos = broker.open_position("EURUSD", "buy", 1.0, 1.10000, 1.09500, 1.11000)
        assert broker.open_count == 1

        trade = broker.close_position(pos.id, 1.10500, reason="tp")
        assert trade is not None
        assert trade.pnl > 0  # 50 pip profit at $10/pip = $500
        assert broker.balance > 10_000
        assert broker.open_count == 0

    def test_open_and_close_loss(self, broker):
        pos = broker.open_position("EURUSD", "sell", 0.5, 1.10000, 1.10500, 1.09000)
        trade = broker.close_position(pos.id, 1.10300, reason="sl")
        assert trade.pnl < 0
        assert broker.balance < 10_000

    def test_sl_tp_check(self, broker):
        broker.open_position("EURUSD", "buy", 1.0, 1.10000, 1.09500, 1.11000)
        # Bar that hits SL
        closed = broker.check_sl_tp("EURUSD", current_high=1.10200, current_low=1.09400)
        assert len(closed) == 1
        assert closed[0].exit_reason == "sl"

    def test_tp_hit(self, broker):
        broker.open_position("EURUSD", "buy", 1.0, 1.10000, 1.09500, 1.10500)
        closed = broker.check_sl_tp("EURUSD", current_high=1.10600, current_low=1.10100)
        assert len(closed) == 1
        assert closed[0].exit_reason == "tp"


class TestPositionSizing:
    def test_lot_size_1pct_risk(self, broker):
        # 1% of 10000 = $100, SL = 50 pips, pip_value = $10 -> 0.20 lots
        lots = broker.calc_lot_size("EURUSD", 1.10000, 1.09500, risk_pct=1.0)
        assert lots in (0.19, 0.20)

    def test_has_position(self, broker):
        assert not broker.has_position("EURUSD")
        broker.open_position("EURUSD", "buy", 0.1, 1.10000, 1.09500, 1.11000)
        assert broker.has_position("EURUSD")


class TestPersistence:
    def test_state_saves_and_loads(self, tmp_path):
        from live.paper_broker import PaperBroker
        sf = str(tmp_path / "persist.json")

        b1 = PaperBroker(initial_balance=5000, state_file=sf)
        b1.open_position("EURUSD", "buy", 0.5, 1.10000, 1.09500, 1.11000)
        b1.close_position(1, 1.10200, "tp")

        # New instance loads state
        b2 = PaperBroker(initial_balance=5000, state_file=sf)
        assert b2.balance == b1.balance
        assert len(b2.history) == 1
        assert b2.history[0].pnl > 0


class TestStats:
    def test_stats_empty(self, broker):
        s = broker.stats()
        assert s["total_trades"] == 0
        assert s["balance"] == 10_000

    def test_stats_after_trades(self, broker):
        broker.open_position("EURUSD", "buy", 1.0, 1.10000, 1.09500, 1.11000)
        broker.close_position(1, 1.10500, "tp")  # win
        broker.open_position("EURUSD", "sell", 1.0, 1.10500, 1.11000, 1.09500)
        broker.close_position(2, 1.11000, "sl")  # loss

        s = broker.stats()
        assert s["total_trades"] == 2
        assert s["winners"] == 1
        assert s["losers"] == 1
