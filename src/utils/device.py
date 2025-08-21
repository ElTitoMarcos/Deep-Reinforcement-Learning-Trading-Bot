import os
import multiprocessing
import time
import logging
import torch


def get_device() -> str:
    """Return the best available torch device and log its details."""
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - hardware/driver issues
            name = "unknown"
        logging.getLogger(__name__).info("Dispositivo: CUDA (%s)", name)
        return "cuda"
    logging.getLogger(__name__).info("Dispositivo: CPU")
    return "cpu"


def sanity_check() -> float:
    """Run a tiny forward/backward pass and log execution time."""
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
    """Limit PyTorch to a subset of CPU threads.

    Parameters
    ----------
    max_threads: int | None
        Optional upper bound for the number of threads. If ``None`` the
        function uses all available cores minus one.
    Returns
    -------
    int
        The number of threads configured via ``torch.set_num_threads``.
    """
    cores = multiprocessing.cpu_count() or 1
    target = max(1, cores - 1)
    if max_threads is not None:
        target = min(target, int(max_threads))
    torch.set_num_threads(target)
    os.environ["OMP_NUM_THREADS"] = str(target)
    return target
