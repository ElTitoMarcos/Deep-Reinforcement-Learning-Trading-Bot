from __future__ import annotations

from typing import Mapping, Any


SYSTEM_PROMPT = "Eres un asistente experto en trading cuantitativo."

PROMPT_PERIODICO = (
    """
Contexto del sistema:
- Objetivo: mejorar el rendimiento económico del bot de trading.
- Datos: {resumen_datos} (ventanas de alta actividad, símbolos, timeframe mínimo, filtros Binance, fees/slippage estimados).
- Entrenamiento actual: algoritmo {algo}, hiperparámetros {hparams}, episodios {episodios}, recompensa media {reward}, PnL valid {pnl}, MaxDD {dd}, Consistencia {cons}.
- Restricciones: no sugieras operaciones en tiempo real; solo ajustes del sistema de entrenamiento/evaluación y selección de símbolos/ventanas.
Tarea:
1) Evalúa si el entrenamiento converge o muestra sobreajuste (da señales concretas).
2) Propón cambios concretos en:
   - símbolos (añadir/quitar, justificar con volatilidad/volumen)
   - ventanas de datos (más o menos “actividad”)
   - hiperparámetros (3 cambios máximos, con motivo)
   - estrategia (PPO vs DQN o híbrido) y pesos del híbrido
3) Devuelve JSON con campos:
   {"symbols_to_add": [], "symbols_to_drop": [], "hparams_patch": {...}, "algo_choice": "ppo|dqn|hybrid", "hybrid_weights": {"det":0.5,"stoch":0.3,"value":0.2}, "rationale": "texto breve"}
Formato: SOLO JSON válido.
"""
).strip()


def build_data_summary(cfg: Mapping[str, Any]) -> str:
    """Return a terse description of the current data context."""

    symbols = ",".join(cfg.get("symbols", []))
    timeframe = cfg.get("timeframe", "1m")
    fees = cfg.get("fees", {})
    filt = cfg.get("filters", {})
    return f"símbolos {symbols}; timeframe {timeframe}; fees {fees}; filtros {filt}"


def build_periodic_prompt(
    cfg: Mapping[str, Any],
    algo: str,
    hparams: Mapping[str, Any],
    episodios: int,
    reward: float,
    pnl: float,
    dd: float,
    cons: float,
) -> str:
    """Fill :data:`PROMPT_PERIODICO` with the provided context."""

    resumen = build_data_summary(cfg)
    return PROMPT_PERIODICO.format(
        resumen_datos=resumen,
        algo=algo,
        hparams=hparams,
        episodios=episodios,
        reward=reward,
        pnl=pnl,
        dd=dd,
        cons=cons,
    )
