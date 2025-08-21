import os
import multiprocessing
import time
import logging
import torch
import shutil


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_badge():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    if shutil.which("nvidia-smi"):
        return "CPU (PyTorch sin CUDA)"
    return "CPU"


def sanity_check() -> float:
    device = get_device()
    x = torch.randn(8, 8, device=device, requires_grad=True)
    start = time.time()
    y = (x * 2).sum()
    y.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - start
    logging.getLogger(__name__).info("sanity_check %.4fs cuda=%s", dt, device == "cuda")
    return dt


def set_cpu_threads(max_threads: int | None = None) -> int:
    cores = multiprocessing.cpu_count() or 1
    target = max(1, cores - 1)
    if max_threads is not None:
        target = min(target, int(max_threads))
    torch.set_num_threads(target)
    os.environ["OMP_NUM_THREADS"] = str(target)
    return target
