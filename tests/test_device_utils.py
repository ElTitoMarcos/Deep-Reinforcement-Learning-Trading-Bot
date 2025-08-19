import torch
from src.utils.device import get_device, set_cpu_threads


def test_get_device_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == "cpu"


def test_get_device_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == "cuda"


def test_set_cpu_threads(monkeypatch):
    # Ensure deterministic core count for the test
    monkeypatch.setattr("src.utils.device.multiprocessing.cpu_count", lambda: 4)
    orig = torch.get_num_threads()
    n = set_cpu_threads(max_threads=2)
    try:
        assert n == 2
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(orig)
