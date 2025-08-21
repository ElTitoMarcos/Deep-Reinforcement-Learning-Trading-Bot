import os
from pathlib import Path


def _get(name: str):
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None  # vacío -> None


def mask(s: str, show: int = 4):
    return "*" * max(0, len(s) - show) + (s[-show:] if s else "")


def load_env():
    cfg = {
        "use_testnet": (_get("BINANCE_USE_TESTNET") or "false").lower() == "true",
        "mainnet_key": _get("BINANCE_API_KEY_MAINNET"),
        "mainnet_sec": _get("BINANCE_API_SECRET_MAINNET"),
        "testnet_key": _get("BINANCE_API_KEY_TESTNET"),
        "testnet_sec": _get("BINANCE_API_SECRET_TESTNET"),
        "openai_key": _get("OPENAI_API_KEY"),
        "default_fee_bps": float(_get("BINANCE_DEFAULT_FEE_BPS") or "10"),
        "rate_limit_per_min": int(_get("RATE_LIMIT_PER_MIN") or "1200"),
    }
    return cfg


def write_env_local(updates: dict, filename: str = ".env.local"):
    p = Path(filename)
    existing: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    existing.update({k: str(v) for k, v in updates.items() if v is not None})
    buf = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    p.write_text(buf, encoding="utf-8")
    return p


def load_binance_creds():
    cfg = load_env()
    use_testnet = cfg["use_testnet"]
    key = cfg["testnet_key" if use_testnet else "mainnet_key"]
    sec = cfg["testnet_sec" if use_testnet else "mainnet_sec"]

    if not key or not sec:
        raise RuntimeError(
            "Credenciales Binance no configuradas. Define BINANCE_API_KEY_/SECRET_ (MAINNET/TESTNET) en .env"
        )
    return key, sec, use_testnet


def load_openai_key():
    cfg = load_env()
    k = cfg.get("openai_key")
    if not k:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")
    return k


def compute_rate_limit_ms():
    cfg = load_env()
    rpm = cfg.get("rate_limit_per_min", 1200)
    return max(1, int(60000 / max(1, rpm)))

