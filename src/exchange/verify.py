def verify_binance_mainnet(ex):
    try:
        acct = ex.private_get_account()
        return True, "Conexión OK (mainnet spot)", {"canTrade": acct.get("canTrade")}
    except Exception as e:
        msg = str(e)
        if "-1021" in msg:
            return False, "Reloj desincronizado (-1021). Se intentó auto-ajuste; reintenta.", {}
        if "-2014" in msg or "-2015" in msg:
            return False, "Clave/permiso inválido (-2014/-2015). Verifica API spot y permisos IP.", {}
        if "403" in msg or "Forbidden" in msg:
            return False, "Acceso denegado (403). Revisa restricciones de IP en Binance.", {}
        return False, f"Error al verificar mainnet: {msg}", {}


def verify_binance_testnet(ex):
    try:
        ex.public_get_ping()
        ex.public_get_exchangeinfo()
        return True, "Conexión OK (testnet spot). 'tradeFee' no disponible: se usará fallback.", {}
    except Exception as e:
        return False, f"Error testnet: {e}", {}


def verify_openai(openai_key: str):
    try:
        if not openai_key or not openai_key.strip():
            return False, "OPENAI_API_KEY vacía.", {}
        return True, "OPENAI_API_KEY presente (verificación mínima).", {}
    except Exception as e:
        return False, f"Error OpenAI: {e}", {}
