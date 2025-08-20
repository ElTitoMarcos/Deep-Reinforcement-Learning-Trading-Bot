import os
import time
import hmac
import hashlib
from pathlib import Path
from typing import Tuple

import requests
import streamlit as st
from dotenv import load_dotenv, set_key


BINANCE_MAINNET = "https://api.binance.com"
BINANCE_TESTNET = "https://testnet.binance.vision"


def _badge(label: str, ok: bool) -> str:
    color = "green" if ok else "red"
    text = "Conectado" if ok else "No conectado"
    return f"{label}: :{color}[{text}]"


def verify_binance_mainnet(api_key: str, api_secret: str) -> Tuple[bool, str]:
    """Ping mainnet and call signed account endpoint."""
    try:
        requests.get(f"{BINANCE_MAINNET}/api/v3/ping", timeout=5).raise_for_status()
        ts = int(time.time() * 1000)
        query = f"timestamp={ts}"
        signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        headers = {"X-MBX-APIKEY": api_key}
        params = {"timestamp": ts, "signature": signature}
        requests.get(f"{BINANCE_MAINNET}/api/v3/account", params=params, headers=headers, timeout=5).raise_for_status()
        return True, ""
    except Exception as e:  # pragma: no cover - network
        return False, str(e)


def verify_binance_testnet(api_key: str, api_secret: str) -> Tuple[bool, str]:
    """Ping testnet and fetch exchangeInfo (no auth required)."""
    try:
        requests.get(f"{BINANCE_TESTNET}/api/v3/ping", timeout=5).raise_for_status()
        requests.get(f"{BINANCE_TESTNET}/api/v3/exchangeInfo", timeout=5).raise_for_status()
        return True, ""
    except Exception as e:  # pragma: no cover - network
        return False, str(e)


def verify_openai(api_key: str) -> Tuple[bool, str]:
    """Minimal OpenAI check using models.list."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover - import
        return False, f"package missing: {e}"
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, ""
    except Exception as e:  # pragma: no cover - network
        return False, str(e)


def _save_env(values: dict, persist: bool) -> None:
    for key, val in values.items():
        if val:
            os.environ[key] = val
    if persist:
        env_path = Path(".env.local")
        env_path.touch(exist_ok=True)
        for key, val in values.items():
            set_key(env_path, key, val or "")
        st.warning("Claves guardadas en .env.local. No lo subas al repositorio.")


def main() -> None:
    st.set_page_config(page_title="Conexiones", layout="wide")
    st.title("🔐 Conexiones")

    load_dotenv(".env.local")
    load_dotenv()

    binance_key = st.text_input("Binance API Key (mainnet)", value=os.getenv("BINANCE_API_KEY", ""))
    binance_secret = st.text_input(
        "Binance API Secret (mainnet)", value=os.getenv("BINANCE_API_SECRET", ""), type="password"
    )
    test_key = st.text_input("Binance API Key (testnet)", value=os.getenv("BINANCE_TESTNET_API_KEY", ""))
    test_secret = st.text_input(
        "Binance API Secret (testnet)", value=os.getenv("BINANCE_TESTNET_API_SECRET", ""), type="password"
    )
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    persist = st.checkbox("Guardar en .env.local", value=False)
    st.caption("Las claves se guardan en memoria; activa el checkbox para persistir en .env.local")

    if st.button("Guardar y verificar"):
        values = {
            "BINANCE_API_KEY": binance_key,
            "BINANCE_API_SECRET": binance_secret,
            "BINANCE_TESTNET_API_KEY": test_key,
            "BINANCE_TESTNET_API_SECRET": test_secret,
            "OPENAI_API_KEY": openai_key,
        }
        _save_env(values, persist)
        b_main, err_main = (False, "")
        b_test, err_test = (False, "")
        oai, err_oai = (False, "")
        if binance_key and binance_secret:
            b_main, err_main = verify_binance_mainnet(binance_key, binance_secret)
        if test_key and test_secret:
            b_test, err_test = verify_binance_testnet(test_key, test_secret)
        if openai_key:
            oai, err_oai = verify_openai(openai_key)
        st.session_state["binance_mainnet_ok"] = b_main
        st.session_state["binance_testnet_ok"] = b_test
        st.session_state["openai_ok"] = oai
        if b_main:
            st.success("Binance mainnet OK")
        elif err_main:
            st.error(f"Binance mainnet: {err_main}")
        if b_test:
            st.success("Binance testnet OK")
        elif err_test:
            st.error(f"Binance testnet: {err_test}")
        if oai:
            st.success("OpenAI OK")
        elif err_oai:
            st.error(f"OpenAI: {err_oai}")

    st.markdown("## Estado")
    col1, col2, col3 = st.columns(3)
    col1.markdown(_badge("Binance mainnet", st.session_state.get("binance_mainnet_ok", False)))
    col2.markdown(_badge("Binance testnet", st.session_state.get("binance_testnet_ok", False)))
    col3.markdown(_badge("OpenAI", st.session_state.get("openai_ok", False)))


if __name__ == "__main__":
    main()
