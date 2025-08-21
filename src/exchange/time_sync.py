import time


def smart_time_sync(ex, retries=3, delay=0.5):
    """Attempt to synchronise time with the exchange."""
    for _ in range(retries):
        try:
            ex.fetch_time()
            return
        except Exception:
            time.sleep(delay)
