from __future__ import annotations
import os
from typing import Any, Dict, List
from pathlib import Path
import copy
import yaml
from dotenv import load_dotenv


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _find_missing(defaults: Dict[str, Any], cfg: Dict[str, Any], prefix: str = "") -> List[str]:
    missing: List[str] = []
    for k, v in defaults.items():
        key = f"{prefix}{k}" if prefix else k
        if k not in cfg:
            missing.append(key)
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            missing.extend(_find_missing(v, cfg[k], key + "."))
    return missing


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML config, merging with defaults and environment variables."""
    load_dotenv()

    defaults_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    with open(defaults_path, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(defaults)
    _deep_update(cfg, user_cfg)
    cfg["_defaults_used"] = _find_missing(defaults, user_cfg)

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

    use_testnet = os.getenv("BINANCE_USE_TESTNET")
    if use_testnet is not None:
        cfg["binance_use_testnet"] = use_testnet.lower() in ("1", "true", "yes")

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
