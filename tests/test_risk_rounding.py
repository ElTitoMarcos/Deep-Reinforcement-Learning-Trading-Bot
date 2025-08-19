import pytest

from src.utils.risk import round_to_step, round_to_tick, passes_min_notional


def test_rounding_to_tick_and_step():
    assert round_to_tick(100.123, 0.05) == pytest.approx(100.1)
    assert round_to_step(1.23456, 0.001) == pytest.approx(1.235)


def test_min_notional_check():
    assert passes_min_notional(100.0, 0.05, 5.0)
    assert not passes_min_notional(100.0, 0.04, 5.0)
