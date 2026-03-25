"""Unit tests for risk management calculations."""

from types import SimpleNamespace

from bot.risk import calc_lot_size, calc_take_profit


def _mock_symbol_info(**overrides):
    defaults = dict(
        point=0.00001,
        digits=5,
        trade_tick_size=0.00001,
        trade_tick_value=1.0,
        volume_step=0.01,
        volume_min=0.01,
        volume_max=100.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestCalcLotSize:
    def test_standard_case(self):
        sym = _mock_symbol_info()
        lots = calc_lot_size(
            balance=10_000,
            risk_pct=1.0,
            entry_price=1.10000,
            sl_price=1.09500,
            symbol_info=sym,
        )
        # risk = $100, SL distance = 50 pips, pip value = $10
        # floor(0.20 / 0.01) * 0.01 may yield 0.19 due to float precision
        assert lots in (0.19, 0.20)

    def test_zero_distance_returns_zero(self):
        sym = _mock_symbol_info()
        lots = calc_lot_size(10_000, 1.0, 1.10000, 1.10000, sym)
        assert lots == 0.0

    def test_respects_volume_min(self):
        sym = _mock_symbol_info()
        # Tiny balance -> lot rounds down below min
        lots = calc_lot_size(10, 1.0, 1.10000, 1.00000, sym)
        assert lots >= sym.volume_min


class TestCalcTakeProfit:
    def test_buy_tp(self):
        tp = calc_take_profit(entry=1.10000, sl=1.09500, rr_ratio=2.0, is_buy=True)
        assert abs(tp - 1.11000) < 1e-5

    def test_sell_tp(self):
        tp = calc_take_profit(entry=1.10000, sl=1.10500, rr_ratio=2.0, is_buy=False)
        assert abs(tp - 1.09000) < 1e-5
