from __future__ import annotations

import logging

from src.reports.human_friendly import episode_sentence
from src.ui import log_stream


def test_episode_sentence_basic() -> None:
    metrics = {"pnl": 0.012, "consistency": 0.6, "turnover": 0.5}
    sent = episode_sentence(metrics)
    assert "+1.2%" in sent
    assert "consistencia media" in sent
    assert "actividad moderada" in sent


def test_log_stream_integration() -> None:
    log_stream._LOG_BUFFER.clear()
    logging.getLogger().info(
        "",
        extra={
            "event": "episode_metrics",
            "metrics": {"pnl": 0.01, "consistency": 0.2, "turnover": 2.0},
        },
    )
    gen = log_stream.subscribe(kinds={"metricas"})
    item = next(gen)
    assert item["kind"] == "metricas"
    assert "+1.0%" in item["message"]
    assert "consistencia baja" in item["message"]
    assert "actividad alta" in item["message"]

