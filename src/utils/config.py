from __future__ import annotations
import os
from typing import Any, Dict
import yaml
from dotenv import load_dotenv


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML config and environment variables (.env) with overrides."""
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Fill from env if present
    env_map = {
        "exchange": os.getenv("EXCHANGE"),
        "api_key": os.getenv("API_KEY"),
        "api_secret": os.getenv("API_SECRET"),
        "api_password": os.getenv("API_PASSWORD"),
    }
    for k, v in env_map.items():
        if v is not None:
            cfg.setdefault("env", {})[k] = v

    # Apply explicit overrides (e.g., CLI flags)
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    return cfg
