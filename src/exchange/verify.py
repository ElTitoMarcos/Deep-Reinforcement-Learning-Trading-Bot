
def verify_private_mainnet(ex):
    try:
        acct = ex.private_get_account()
        return True, "Conexión OK (mainnet spot)", {"canTrade": acct.get("canTrade", None)}
    except Exception as e:
        msg = str(e)
        if "-1021" in msg:
            return False, "Reloj desincronizado (-1021). Se autoajustó; reintenta.", {}
        if "-2014" in msg or "-2015" in msg:
            return False, "Clave/permiso inválido (-2014/-2015). Revisa permisos SPOT o la IP en Binance.", {}
        if "403" in msg:
            return False, "Acceso denegado (403). Comprueba restricción de IP en Binance.", {}
        return False, f"Error mainnet: {msg}", {}


def verify_private_testnet(ex):
    try:
        ex.public_get_ping()
        ex.public_get_exchangeinfo()
        return True, "OK (testnet). 'tradeFee' no existe; se usará fallback.", {}
    except Exception as e:
        return False, f"Error testnet: {e}", {}
