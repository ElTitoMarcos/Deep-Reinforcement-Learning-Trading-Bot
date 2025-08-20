import json
from src.auto.timestep_planner import plan_timesteps


class DummyLLM:
    def __init__(self, factor: float = 1.2):
        self.factor = factor

    def ask(self, system: str, prompt: str) -> str:  # pragma: no cover - simple stub
        return json.dumps({"factor": self.factor})


def test_warmup_stage():
    runs: list[int] = []
    ts = plan_timesteps({"stage": "warmup"}, 0.0, {}, runs)
    assert ts == 10_000
    assert runs == [10_000]
    assert "fase inicial" in plan_timesteps.last_reason


def test_consolidation_long_with_stability():
    runs: list[int] = []
    stability = {"score_trend": 0.5, "td_var": 0.01}
    ts = plan_timesteps({"stage": "consolidation"}, 0.8, stability, runs)
    assert ts >= 200_000
    assert "score" in plan_timesteps.last_reason


def test_llm_adjustment():
    runs: list[int] = []
    ts = plan_timesteps({"stage": "warmup"}, 0.0, {}, runs, llm=DummyLLM(1.25))
    assert ts == int(10_000 * 1.25)
    assert "ajuste LLM" in plan_timesteps.last_reason
