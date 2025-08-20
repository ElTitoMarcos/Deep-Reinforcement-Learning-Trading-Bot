import json
import random
from pathlib import Path

from src.auto import RewardTuner


def _ensure_bandit_valid(rt: RewardTuner) -> None:
    """Adjust bandit arms to avoid zero-variance beta draws."""
    for arms in rt.bandit.values():
        for arm in arms.values():
            arm.trials = arm.success + 1


def test_propose_and_confirm(tmp_path):
    random.seed(0)
    mem = tmp_path / "mem.jsonl"
    tuner = RewardTuner({"w_pnl": 1.0}, {"w_pnl": (0.5, 2.0)}, mem)
    _ensure_bandit_valid(tuner)

    before = dict(tuner.weights)
    proposal = tuner.propose({"pnl": 1.0})
    assert proposal["w_pnl"] != before["w_pnl"]

    tuner.confirm({"pnl": 2.0})
    assert tuner.weights["w_pnl"] == proposal["w_pnl"]
    records = [json.loads(line) for line in mem.read_text().splitlines()]
    assert records[-1]["success"] is True

    _ensure_bandit_valid(tuner)
    prev = dict(tuner.weights)
    tuner.propose({"pnl": 1.0})
    tuner.confirm({"pnl": -1.0})
    assert tuner.weights["w_pnl"] == prev["w_pnl"]
    records = [json.loads(line) for line in mem.read_text().splitlines()]
    assert records[-1]["success"] is False
