from __future__ import annotations
import os, sys, json, time
from typing import Any, Dict

class JsonLogger:
    def __init__(self, path: str | None = None):
        self.path = path
        self.fp = open(path, "a", encoding="utf-8") if path else None

    def log(self, level: str, msg: str, **kwargs: Any):
        rec = {"ts": time.time(), "level": level, "msg": msg}
        rec.update(kwargs)
        line = json.dumps(rec, ensure_ascii=False)
        if self.fp:
            self.fp.write(line + "\n")
            self.fp.flush()
        else:
            print(line, file=sys.stdout)

    # convenience wrappers -------------------------------------------------

    def info(self, msg: str, **kwargs: Any):
        self.log("info", msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any):
        self.log("warning", msg, **kwargs)

    def error(self, msg: str, **kwargs: Any):
        self.log("error", msg, **kwargs)

    # context manager support ----------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self.fp:
            self.fp.close()

def ensure_logger(path: str | None) -> JsonLogger:
    os.makedirs(os.path.dirname(path), exist_ok=True) if path else None
    return JsonLogger(path)
