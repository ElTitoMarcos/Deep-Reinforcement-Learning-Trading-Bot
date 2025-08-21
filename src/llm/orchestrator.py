from __future__ import annotations

"""LLM-based auto tuning orchestrator.

This module glues together specialised advisors that each focus on a
particular portion of the global configuration.  The orchestrator calls the
LLM for suggestions, decides which ones to apply and records the result so that
future decisions can use the historic context.
"""

from dataclasses import dataclass
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from .client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advisor registry

@dataclass
class _Advisor:
    prompt_builder: Callable[[Mapping[str, Any]], tuple[str, str]]
    parser: Callable[[str], Mapping[str, Any]]
    impact_weight: float = 1.0


class AdvisorRegistry:
    """Registry of specialised advisors and helpers to run them."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._advisors: Dict[str, _Advisor] = {}

    def register(
        self,
        name: str,
        prompt_builder: Callable[[Mapping[str, Any]], tuple[str, str]],
        parser: Callable[[str], Mapping[str, Any]],
        impact_weight: float = 1.0,
    ) -> None:
        """Register a new advisor.

        ``prompt_builder`` should return ``(system_prompt, user_prompt)``. The
        parser is expected to return a mapping with at least the keys
        ``changes`` (a ``dict`` of proposed key/value pairs), ``rationale`` and
        ``confidence`` (``0-1``).
        """

        self._advisors[name] = _Advisor(prompt_builder, parser, impact_weight)

    # ------------------------------------------------------------------
    def run_all(self, context: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all registered advisors and collect proposals.

        The returned mapping uses the proposed config key as first level key and
        contains the proposed value together with metadata about the origin of
        the proposal.
        """

        proposals: Dict[str, Dict[str, Any]] = {}
        for name, adv in self._advisors.items():
            try:
                sys_prompt, user_prompt = adv.prompt_builder(context)
                raw = self.client.ask(sys_prompt, user_prompt)
                parsed = adv.parser(raw)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("advisor_failed", advisor=name, error=str(exc))
                continue

            for key, value in parsed.get("changes", {}).items():
                proposals[key] = {
                    "value": value,
                    "rationale": parsed.get("rationale", ""),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "impact": float(parsed.get("impact", 0.0)) * adv.impact_weight,
                    "advisor": name,
                }
        return proposals


# ---------------------------------------------------------------------------
# Frequency controller

class FrequencyController:
    """Simple controller that adapts the next call cadence based on impact."""

    def __init__(
        self,
        *,
        min_interval: float = 300.0,
        max_interval: float = 7200.0,
        init_interval: float = 1800.0,
        positive_threshold: float = 0.0,
    ) -> None:
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.interval = float(init_interval)
        self.positive_threshold = float(positive_threshold)

    def update(self, impact_score: float) -> float:
        """Update cadence based on ``impact_score`` and return new interval."""

        if impact_score > self.positive_threshold:
            # Improvements → call more frequently
            self.interval = max(self.min_interval, self.interval * 0.5)
        elif impact_score < 0:
            # Negative impact → back off
            self.interval = min(self.max_interval, self.interval * 1.5)
        # else keep interval
        return self.interval


# ---------------------------------------------------------------------------
# Orchestrator

class LLMOrchestrator:
    """High level component coordinating advisor runs and applying patches."""

    def __init__(
        self,
        registry: AdvisorRegistry,
        freq_ctrl: FrequencyController,
        *,
        report_dir: Path | str = "reports/llm_runs",
        confidence_threshold: float = 0.6,
        impact_threshold: float = 0.0,
    ) -> None:
        self.registry = registry
        self.freq_ctrl = freq_ctrl
        self.confidence_threshold = confidence_threshold
        self.impact_threshold = impact_threshold
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = int(time.time())

    # ------------------------------------------------------------------
    def step(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one orchestration step.

        ``run_context`` should contain at least ``config`` (a dict), ``metrics``
        and ``ui`` (optional object with ``notify`` method).  The method returns
        the dictionary of accepted proposals.
        """

        proposals = self.registry.run_all(run_context)
        accepted: Dict[str, Dict[str, Any]] = {}
        impact_score = 0.0

        config = run_context.get("config", {})
        for key, prop in proposals.items():
            if prop["confidence"] < self.confidence_threshold:
                continue
            if prop.get("impact", 0.0) < self.impact_threshold:
                continue
            config[key] = prop["value"]
            accepted[key] = prop
            impact_score += prop.get("impact", 0.0)

        # notify UI if available
        ui = run_context.get("ui")
        if ui and hasattr(ui, "notify"):
            try:  # pragma: no cover - UI side effects
                ui.notify("config_updated", accepted)
            except Exception:
                logger.warning("ui_notify_failed", exc_info=True)

        record = {
            "timestamp": time.time(),
            "context": run_context,
            "proposals": proposals,
            "accepted": accepted,
            "impact_score": impact_score,
        }
        report_file = self.report_dir / f"{self.run_id}.jsonl"
        with report_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        # update cadence for next run
        self.freq_ctrl.update(impact_score)
        return accepted


# ---------------------------------------------------------------------------
# Example advisor implementations


def _json_parser(text: str) -> Mapping[str, Any]:
    """Parse JSON responses from the LLM."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"changes": {}, "rationale": text, "confidence": 0.0}


def reward_prompt(ctx: Mapping[str, Any]) -> tuple[str, str]:
    """Build prompt for reward weight advisor."""

    metrics = ctx.get("metrics", {})
    cfg = ctx.get("config", {})
    system = "Eres un asistente que ajusta pesos de recompensas."
    user = json.dumps({"metrics": metrics, "weights": {k: cfg.get(k) for k in ["w_pnl", "w_drawdown", "w_volatility", "w_turnover"]}})
    return system, user


def window_prompt(ctx: Mapping[str, Any]) -> tuple[str, str]:
    system = "Asistente de ventanas de datos"  # simple placeholder
    user = json.dumps({"vol_window": ctx.get("config", {}).get("vol_window"), "act_window": ctx.get("config", {}).get("act_window")})
    return system, user


def algo_prompt(ctx: Mapping[str, Any]) -> tuple[str, str]:
    system = "Asistente de mezcla algorítmica"
    user = json.dumps({"weights": ctx.get("config", {}).get("algo_weights")})
    return system, user


def hparam_prompt(ctx: Mapping[str, Any]) -> tuple[str, str]:
    system = "Asistente de hiperparámetros"
    hp = {k: ctx.get("config", {}).get(k) for k in ("lr", "batch_size", "horizon")}
    user = json.dumps(hp)
    return system, user


def build_default_registry(client: LLMClient) -> AdvisorRegistry:
    """Return a registry pre-populated with the standard advisors."""

    reg = AdvisorRegistry(client)
    reg.register("reward", reward_prompt, _json_parser, impact_weight=1.0)
    reg.register("windows", window_prompt, _json_parser, impact_weight=0.5)
    reg.register("algo_mix", algo_prompt, _json_parser, impact_weight=0.8)
    reg.register("hparams", hparam_prompt, _json_parser, impact_weight=1.2)
    return reg


__all__ = [
    "AdvisorRegistry",
    "FrequencyController",
    "LLMOrchestrator",
    "build_default_registry",
]
