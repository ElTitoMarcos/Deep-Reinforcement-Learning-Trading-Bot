import time

from src.utils.health import memory_guard, time_guard


def test_memory_guard():
    assert memory_guard(max_gb=1000) is True
    assert memory_guard(max_gb=0) is False


def test_time_guard():
    now = time.time()
    assert time_guard(now, timeout_min=1) is True
    past = now - 120
    assert time_guard(past, timeout_min=1) is False
