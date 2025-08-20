from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict


class StageScheduler:
    """Track training metrics and manage stage transitions.

    Stages progress sequentially: warmup -> exploration -> consolidation -> fine-tune.
    Each stage controls whether the meta-controller and reward tuner may adapt.
    """

    def __init__(
        self,
        thresholds: Dict[str, float] | None = None,
        llm: Any | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.stage_order = ["warmup", "exploration", "consolidation", "fine-tune"]
        self.stage = self.stage_order[0]
        self.llm = llm
        self.timeout = timeout
        self.thresholds = thresholds or {}
        self.score_hist: deque[float] = deque(maxlen=5)
        # rules for allowing adjustments in each stage
        self.rules: Dict[str, Dict[str, bool]] = {
            "warmup": {"allow_tuner": False, "allow_mapping": False},
            "exploration": {"allow_tuner": True, "allow_mapping": True},
            "consolidation": {"allow_tuner": True, "allow_mapping": True},
            "fine-tune": {"allow_tuner": False, "allow_mapping": False},
        }

    # ------------------------------------------------------------------
    def _next_stage(self) -> str | None:
        idx = self.stage_order.index(self.stage)
        if idx + 1 < len(self.stage_order):
            return self.stage_order[idx + 1]
        return None

    # ------------------------------------------------------------------
    def _ask_llm(self, summary: Dict[str, Any]) -> tuple[bool, str]:
        """Ask LLM to approve stage change. Fallback to approve on timeout."""
        if self.llm is None:
            return True, ""
        prompt = json.dumps(summary)
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self.llm.ask, "Stage scheduler", prompt)
                resp = fut.result(timeout=self.timeout)
            data = json.loads(resp)
            return bool(data.get("approve", True)), data.get("notes", "")
        except Exception:
            # timeout or parsing error -> follow heuristic
            return True, "timeout"

    # ------------------------------------------------------------------
    def on_tick(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update scheduler with latest metrics and maybe advance stage."""
        score = float(metrics.get("score", 0.0))
        self.score_hist.append(score)

        reason_parts: list[str] = []
        sustained = False
        if len(self.score_hist) == self.score_hist.maxlen:
            delta = self.score_hist[-1] - self.score_hist[0]
            thr = self.thresholds.get("delta_score", 0.0)
            if delta > thr:
                sustained = True
                reason_parts.append(f"Δscore>{thr}")
        td_ok = metrics.get("td_error", 0.0) <= self.thresholds.get("td_error", float("inf"))
        if td_ok:
            reason_parts.append("TD-error estable")
        dd_ok = metrics.get("drawdown", 0.0) <= self.thresholds.get("drawdown", float("inf"))
        if dd_ok:
            reason_parts.append("DD bajo")
        tr_ok = metrics.get("trade_ratio", 0.0) <= self.thresholds.get("trade_ratio", float("inf"))
        if tr_ok:
            reason_parts.append("ratio trades ok")

        changed = False
        if sustained and td_ok and dd_ok and tr_ok:
            next_stage = self._next_stage()
            if next_stage:
                approved, notes = self._ask_llm({"from": self.stage, "to": next_stage, "metrics": metrics})
                if approved:
                    prev = self.stage
                    self.stage = next_stage
                    changed = True
                    if notes:
                        reason_parts.append(notes)
                    reason = ", ".join(reason_parts)
                    info = {"stage": self.stage, "prev": prev, "changed": True, "reason": reason}
                else:
                    info = {"stage": self.stage, "changed": False}
            else:
                info = {"stage": self.stage, "changed": False}
        else:
            info = {"stage": self.stage, "changed": False}

        rule = self.rules[self.stage]
        info.update(rule)
        return info
