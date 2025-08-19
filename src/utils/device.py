import os
import multiprocessing
import torch


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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
