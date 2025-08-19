import os
import torch


def get_device() -> str:
    """Return "cuda" if an RTX 3070 GPU is available, else "cpu".

    This helper prefers a specific GPU model to ensure reproducibility.  If
    CUDA is available but the device name does not contain "3070", we default
    to CPU to avoid unexpected slower or unsupported GPUs.
    """
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            if "3070" in name:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def set_cpu_threads() -> int:
    """Limit PyTorch to ``n_cores - 1`` threads when running on CPU.

    Returns the number of threads configured.  If CUDA is available this is a
    no-op and the current thread count is returned.
    """
    if get_device() == "cpu":
        n = max(1, (os.cpu_count() or 1) - 1)
        try:
            torch.set_num_threads(n)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = str(n)
        return n
    return torch.get_num_threads()
