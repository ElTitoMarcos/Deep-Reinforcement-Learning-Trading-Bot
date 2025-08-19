from __future__ import annotations
import os
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config and environment variables (.env)."""
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
    # Only add non-None values
    for k, v in env_map.items():
        if v is not None:
            cfg.setdefault("env", {})[k] = v
    return cfg
