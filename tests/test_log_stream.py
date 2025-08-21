import logging

from src.reports.human_friendly import episode_sentence
from src.ui import log_stream
from src.ui.log_stream import get_auto_profile, recent_counts


def test_episode_sentence_basic() -> None:
    metrics = {"pnl": 0.012, "consistency": 0.6, "turnover": 0.5}
    sent = episode_sentence(metrics)
    assert "+1.2%" in sent
    assert "consistencia media" in sent
    assert "actividad moderada" in sent


def test_log_stream_integration() -> None:
    log_stream._LOG_BUFFER.clear()
    log_stream._KIND_BUFFERS.clear()
    logging.getLogger().setLevel(logging.INFO)
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


def test_get_auto_profile_training() -> None:
    prof = get_auto_profile("warmup")
    assert prof == {"reward_tuner", "dqn_stability", "checkpoints"}


def test_get_auto_profile_evaluation() -> None:
    assert get_auto_profile("evaluation") == {"hybrid_weights", "performance"}


def test_get_auto_profile_data() -> None:
    assert get_auto_profile("data") == {"incremental_update", "qc"}


def test_per_kind_limit() -> None:
    log_stream._LOG_BUFFER.clear()
    log_stream._KIND_BUFFERS.clear()
    logger = logging.getLogger()
    for _ in range(250):
        logger.info("test", extra={"kind": "datos"})
    assert len(log_stream._KIND_BUFFERS["datos"]) == 200
    assert sum(1 for e in log_stream._LOG_BUFFER if e["kind"] == "datos") == 200


def test_slippage_coalescing() -> None:
    log_stream._LOG_BUFFER.clear()
    log_stream._KIND_BUFFERS.clear()
    logger = logging.getLogger()
    for i in range(20):
        logger.info("", extra={"event": "slippage_est", "pct": 0.0003 + i * 0.0001, "usd": 10})
    events = [e for e in log_stream._LOG_BUFFER if e["kind"] == "datos"]
    assert len(events) == 1
    msg = events[0]["message"]
    assert "slippage_est x20" in msg
    assert "rango" in msg


def test_recent_counts() -> None:
    log_stream._LOG_BUFFER.clear()
    log_stream._KIND_BUFFERS.clear()
    logger = logging.getLogger()
    logger.info("a", extra={"kind": "trades"})
    logger.info("b", extra={"kind": "riesgo"})
    total, counts = recent_counts(30)
    assert total == 2
    assert counts["trades"] == 1
    assert counts["riesgo"] == 1

