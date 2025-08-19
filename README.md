# DRL Trading Base

Base de proyecto en **Python** para trading algorítmico preparado para **Deep Reinforcement Learning (DRL)**.

## Estructura

```
.
├── configs/
│   └── default.yaml
├── notebooks/
├── scripts/
│   ├── build_universe.py
│   └── download_history.py
├── src/
│   ├── backtest/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── simulator.py
│   ├── data/
│   │   └── ccxt_loader.py
│   ├── env/
│   │   └── trading_env.py
│   ├── policies/
│   │   ├── deterministic.py
│   │   ├── router.py
│   │   ├── stochastic.py
│   │   └── value_based.py
│   ├── training/
│   │   └── train_drl.py
│   └── utils/
│       ├── config.py
│       ├── data_io.py
│       ├── logging.py
│       ├── orderbook.py
│       └── risk.py
├── tests/
│   ├── test_data_loader_smoke.py
│   ├── test_env_smoke.py
│   └── test_policy_router.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e .
```

> **Opcional**: para usar PPO con *stable-baselines3* necesitas `torch`. En CPU basta:
```bash
pip install "torch>=2.2" "stable-baselines3>=2.3.2"
```

## Configuración

Copia `.env.example` a `.env` y ajusta tus claves si vas a descargar datos:
```bash
cp .env.example .env
```

Edita `configs/default.yaml` para símbolos, fees, slippage, hiperparámetros y pesos de *reward*.

## Descarga de datos

```bash
python scripts/download_history.py --exchange binance --symbols BTC/USDT ETH/USDT --timeframe 1m --since "2024-01-01"
```

Los datos se guardan en `data/raw/{exchange}/{symbol}/{timeframe}.parquet`.

## Construir universo de símbolos líquidos

```bash
python scripts/build_universe.py --exchange binance --quote USDT --min-vol-usd 1000000
```

Salida: `data/universe/liquid_universe.csv`.

## Entrenamiento (DRL)

```bash
python -m src.training.train_drl --config configs/default.yaml --algo ppo --timesteps 20000
```

Si no tienes `stable-baselines3`, el script usa una implementación mínima de DQN incluida.

## Backtest y evaluación

```bash
python -m src.backtest.evaluate --config configs/default.yaml --policy deterministic
```

Resultados en `reports/` + métricas en consola.

## Experimento completo

```bash
python scripts/run_experiment.py --config configs/default.yaml --seed 1 --algo dqn --timesteps 1000 --data-mode price_only
```

Descarga datos si faltan, entrena y ejecuta un backtest con reporte en `reports/{exp_id}/`.

## Tests (smoke)

```bash
pytest -q
```

## Troubleshooting: rate limits/timeouts

- **Descarga lenta o errores `RateLimitExceeded`**: incrementa `--rate-limit` y `--retries`.

```bash
python scripts/download_history.py --exchange binance --symbols BTC/USDT --timeframe 1m --rate-limit 2 --retries 10
```

- **`RequestTimeout` en el exchange**: reintenta tras unos segundos; `ccxt` aplica *backoff* exponencial.

## Notas

- El *env* sigue la API Gymnasium (cae a Gym si no está disponible).
- La observación incluye precio, volatilidad, drawdown local, señal de “murallas” (placeholder), estado de posición y trailing.
- Acciones discretas: 0=hold, 1=open/long, 2=close. Hay *hooks* para acciones continuas.
- *Reward* modular con *heads* configurables: PnL, turnover, drawdown y volatilidad.
- *Experience buffer* y *logging* en JSONL.

¡A iterar sin miedo y con *stop-loss*! 💡

## UI de Configuración (Streamlit)

Instala dependencias y lanza la interfaz:
```bash
pip install -e .  # asegura que streamlit esté instalado por pyproject
python scripts/run_ui.py
# o directamente:
streamlit run src/ui/app.py
```

La UI permite:
- Editar y guardar `configs/default.yaml`
- Descargar histórico (ccxt) y simular 1s desde 1m si no está disponible
- Lanzar entrenamiento (PPO/DQN) y ver la salida
- Evaluar/backtestear con una política
