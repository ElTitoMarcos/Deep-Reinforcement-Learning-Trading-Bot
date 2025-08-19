import pytest

from src.utils.risk import apply_price_tick, apply_qty_step, respects_min_notional


def test_apply_price_tick():
    assert apply_price_tick(100.123, 0.05) == pytest.approx(100.1)
    assert apply_price_tick(100.149, 0.05) == pytest.approx(100.1)
    assert apply_price_tick(100.15, 0.05) == pytest.approx(100.15)


def test_apply_qty_step():
    assert apply_qty_step(1.23456, 0.001) == pytest.approx(1.234)
    assert apply_qty_step(0.98765, 0.01) == pytest.approx(0.98)


def test_respects_min_notional():
    assert respects_min_notional(100.0, 0.05, 5.0)
    assert not respects_min_notional(100.0, 0.049, 5.0)
