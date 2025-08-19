"""Helpers for Binance account metadata."""
from __future__ import annotations

import os
import time
import hmac
import hashlib
import logging
from typing import Dict

import requests
import yaml


logger = logging.getLogger(__name__)


class BinanceMeta:
    """Lightweight client to retrieve account trade fees."""

    def __init__(self, api_key: str | None, api_secret: str | None, use_testnet: bool = False):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.use_testnet = use_testnet
        self._filters_cache: Dict[str, Dict[str, float]] = {}
        self.last_fee_origin = ""

    def get_account_trade_fees(self) -> Dict[str, Dict[str, float]]:
        """Return maker/taker fees per symbol.

        Attempts to query ``/sapi/v1/asset/tradeFee``. On failure uses a
        configurable fallback fee.
        """

        base = "https://testnet.binance.vision" if self.use_testnet else "https://api.binance.com"
        endpoint = "/sapi/v1/asset/tradeFee"
        params = {"timestamp": int(time.time() * 1000)}
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            resp = requests.get(base + endpoint, params=params, headers=headers, timeout=10)
            if self.use_testnet and getattr(resp, "status_code", None) == 404:
                raise requests.HTTPError("testnet tradeFee unsupported", response=resp)
            resp.raise_for_status()
            data = resp.json()
            fees: Dict[str, Dict[str, float]] = {}
            for item in data:
                symbol = item.get("symbol")
                fees[symbol] = {
                    "maker": float(item.get("makerCommission", 0.0)),
                    "taker": float(item.get("takerCommission", 0.0)),
                }
            if fees:
                self.last_fee_origin = "Detectado por API"
                return fees
        except Exception as exc:  # pragma: no cover - network or auth issues
            bps_env = os.getenv("BINANCE_DEFAULT_FEE_BPS")
            if bps_env:
                try:
                    bps = float(bps_env)
                except ValueError:
                    bps = 10.0
                fee = bps / 10000.0
            else:
                try:
                    with open("configs/default.yaml", "r", encoding="utf-8") as fh:
                        cfg = yaml.safe_load(fh)
                    fee = float(cfg.get("fees", {}).get("taker", 0.001))
                    bps = fee * 10000.0
                except Exception:
                    fee = 0.001
                    bps = 10.0
            if self.use_testnet:
                self.last_fee_origin = "Fallback (testnet)"
                logger.warning("testnet no soporta tradeFee; usando fallback %s bps", bps)
            else:
                self.last_fee_origin = "Fallback"
                logger.warning("tradeFee API failed: %s; usando fallback %s bps", exc, bps)
            return {"SPOT": {"maker": fee, "taker": fee}}

    def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        """Return price/lot/minNotional filters for *symbol*.

        Results are cached per symbol to avoid repeatedly querying the
        ``/api/v3/exchangeInfo`` endpoint.
        """

        if symbol in self._filters_cache:
            return self._filters_cache[symbol]

        base = "https://testnet.binance.vision" if self.use_testnet else "https://api.binance.com"
        endpoint = "/api/v3/exchangeInfo"
        params = {"symbol": symbol.replace("/", "")}

        try:
            resp = requests.get(base + endpoint, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("symbols", [])
            if data:
                filt = {}
                for f in data[0].get("filters", []):
                    ft = f.get("filterType")
                    if ft == "PRICE_FILTER":
                        filt["tickSize"] = float(f.get("tickSize", 0.0))
                    elif ft == "LOT_SIZE":
                        filt["stepSize"] = float(f.get("stepSize", 0.0))
                    elif ft == "MIN_NOTIONAL":
                        filt["minNotional"] = float(f.get("minNotional", 0.0))
                if filt:
                    self._filters_cache[symbol] = filt
                    return filt
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("exchangeInfo API failed: %s", exc)

        # fall back to config defaults
        try:
            with open("configs/default.yaml", "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            f = cfg.get("filters", {})
            fallback = {
                "tickSize": float(f.get("tickSize", 0.0)),
                "stepSize": float(f.get("stepSize", 1.0)),
                "minNotional": float(cfg.get("min_notional_usd", 0.0)),
            }
            self._filters_cache[symbol] = fallback
            return fallback
        except Exception:
            fallback = {"tickSize": 0.0, "stepSize": 1.0, "minNotional": 0.0}
            self._filters_cache[symbol] = fallback
            return fallback
