import pandas as pd

from src.data.microv5_features import (
    detectar_resistencia_asks,
    umbral_entrada_dinamico,
    umbral_salida_dinamico,
    derivados_basicos,
)


def test_resistencia_detection():
    asks = [[100, 1], [101, 1], [102, 10]]  # 10 is >=5x mean (which is 4)
    assert detectar_resistencia_asks(asks) == 102


def test_umbral_entrada_dinamico_decreases():
    window = [100, 99.5]  # drop 0.5%
    thr = 0.005
    assert umbral_entrada_dinamico(window, thr, None) < thr


def test_umbral_salida_dinamico_decreases():
    window = [100, 101.5]  # rise 1.5%
    thr = 0.01
    assert umbral_salida_dinamico(window, thr, None) < thr


def test_derivados_basicos():
    ob = {"bids": [[99, 2]], "asks": [[101, 1]]}
    d = derivados_basicos(ob)
    assert d["best_bid"] == 99
    assert d["best_ask"] == 101
    assert d["spread"] == 2
    assert d["mid"] == 100
