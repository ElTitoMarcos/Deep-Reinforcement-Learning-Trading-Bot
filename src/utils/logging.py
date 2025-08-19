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

    def close(self):
        if self.fp:
            self.fp.close()

def ensure_logger(path: str | None) -> JsonLogger:
    os.makedirs(os.path.dirname(path), exist_ok=True) if path else None
    return JsonLogger(path)
