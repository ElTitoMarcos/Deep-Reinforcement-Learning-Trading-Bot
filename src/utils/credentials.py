import os

def _get(name: str) -> str | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None  # vacío -> None

def load_binance_creds():
    use_testnet = (_get("BINANCE_USE_TESTNET") or "false").lower() == "true"

    # Esquema recomendado
    key = _get("BINANCE_API_KEY_TESTNET" if use_testnet else "BINANCE_API_KEY_MAINNET")
    sec = _get("BINANCE_API_SECRET_TESTNET" if use_testnet else "BINANCE_API_SECRET_MAINNET")

    # Compatibilidad con nombres genéricos (por si existen)
    if not key:
        key = _get("API_KEY")
    if not sec:
        sec = _get("API_SECRET")

    if not key or not sec:
        raise RuntimeError(
            "Credenciales Binance no configuradas. Define "
            "BINANCE_API_KEY_/SECRET_ (MAINNET/TESTNET) o API_KEY/API_SECRET en .env"
        )
    return key, sec, use_testnet

def load_openai_key():
    k = _get("OPENAI_API_KEY")
    if not k:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")
    return k

def compute_rate_limit_ms():
    try:
        rpm = int(_get("RATE_LIMIT_PER_MIN") or "1200")
    except ValueError:
        rpm = 1200
    return max(1, int(60000 / max(1, rpm)))

def mask(s: str, show: int = 4) -> str:
    return "*" * max(0, len(s) - show) + (s[-show:] if s else "")
