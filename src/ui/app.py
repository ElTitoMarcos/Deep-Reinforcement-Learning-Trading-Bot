import os, io, sys, json, subprocess, time
from datetime import datetime
import streamlit as st

from src.utils.config import load_config
from src.utils.paths import ensure_dirs_exist, get_raw_dir
from src.data.ccxt_loader import get_exchange, fetch_ohlcv, save_history
from src.data.volatility_windows import find_high_activity_windows
from src.data.symbol_discovery import discover_symbols
from src.exchange.binance_meta import BinanceMeta
from dotenv import load_dotenv
from src.auto.strategy_selector import choose_algo
from src.auto.hparam_tuner import tune

CONFIG_PATH = st.session_state.get("config_path", "configs/default.yaml")

st.set_page_config(page_title="DRL Trading Config", layout="wide")

st.title("⚙️ Configuración DRL Trading")

with st.sidebar:
    st.header("Ajustes globales")
    CONFIG_PATH = st.text_input("Ruta config YAML", value=CONFIG_PATH, key="cfg_path", help="Normalmente configs/default.yaml")
    st.session_state["config_path"] = CONFIG_PATH
    # Cargar YAML
    try:
        cfg = load_config(CONFIG_PATH)
        ensure_dirs_exist(cfg)
    except Exception as e:
        st.error(f"No se pudo cargar {CONFIG_PATH}: {e}")
        cfg = {}

    paths_cfg = cfg.get("paths", {})
    raw_dir = get_raw_dir(cfg)
    use_testnet_default = bool(cfg.get("binance_use_testnet", False))
    mode = st.radio("Modo", ["Mainnet", "Testnet"], index=1 if use_testnet_default else 0)
    use_testnet = mode == "Testnet"
    os.environ["BINANCE_USE_TESTNET"] = "true" if use_testnet else "false"
    st.caption("Símbolos sugeridos (auto)")
    refresh_syms = st.button("Actualizar", key="refresh_syms")
    if "symbol_checks" not in st.session_state or refresh_syms:
        try:
            ex = get_exchange(use_testnet=use_testnet)
            suggested = discover_symbols(ex, top_n=20)
        except Exception as e:
            st.warning(f"Descubrimiento falló: {e}")
            suggested = cfg.get("symbols") or ["BTC/USDT"]
        checks = st.session_state.get("symbol_checks", {})
        for s in suggested:
            checks.setdefault(s, True)
        st.session_state["symbol_checks"] = checks
    checks = st.session_state.get("symbol_checks", {})
    for sym in sorted(checks):
        checks[sym] = st.checkbox(sym, value=checks[sym], key=f"sym_{sym}")
    manual = st.text_input("Añadir manualmente", key="manual_sym").upper().strip()
    if manual and manual not in checks:
        checks[manual] = True
    selected_symbols = [s for s, v in checks.items() if v]
    cfg["symbols"] = selected_symbols

    fees_dict = cfg.get("fees", {})
    fees_maker = st.number_input("Fee maker", value=float(fees_dict.get("maker",0.001)), step=0.0001, format="%.6f", key="fee_maker")
    fees_taker = st.number_input("Fee taker", value=float(fees_dict.get("taker",0.001)), step=0.0001, format="%.6f", key="fee_taker")
    if st.button("Actualizar comisiones"):
        load_dotenv()
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        try:
            meta = BinanceMeta(api_key, api_secret, use_testnet)
            fee_map = meta.get_account_trade_fees()
            symbol_key = (selected_symbols[0].replace("/", "") if selected_symbols else next(iter(fee_map)))
            entry = fee_map.get(symbol_key) or next(iter(fee_map.values()))
            st.session_state["fee_maker"] = entry.get("maker", fees_maker)
            st.session_state["fee_taker"] = entry.get("taker", fees_taker)
            st.success(f"Maker {entry.get('maker',0)} | Taker {entry.get('taker',0)}")
        except Exception as e:
            st.error(f"No se pudo obtener: {e}")
    fees_maker = st.session_state.get("fee_maker", fees_maker)
    fees_taker = st.session_state.get("fee_taker", fees_taker)
    cfg["fees"] = {"maker": fees_maker, "taker": fees_taker}
    slippage_mult = st.number_input(
        "Multiplicador slippage",
        value=float(cfg.get("slippage_multiplier", 1.0)),
        step=0.1,
        format="%.2f",
    )
    min_notional = st.number_input("Mínimo notional USD", value=float(cfg.get("min_notional_usd",10.0)), step=1.0)

    tick_size = st.number_input("tickSize", value=float(cfg.get("filters",{}).get("tickSize",0.01)))
    step_size = st.number_input("stepSize", value=float(cfg.get("filters",{}).get("stepSize",0.0001)))

    st.caption("Reward heads (pesos)")
    rw = cfg.get("reward_weights", {"pnl": 1.0, "turn": 0.1, "dd": 0.2, "vol": 0.1})
    beneficio = st.number_input(
        "Beneficio (más alto = priorizar ganar dinero)",
        value=float(rw.get("pnl", 1.0)),
        help="Sube si buscas ganancias; sugerido 1.0",
        key="w_pnl",
    )
    control_act = st.number_input(
        "Control de actividad (más alto = operar menos)",
        value=float(rw.get("turn", 0.1)),
        help="Sube para operar menos; sugerido 0.1",
        key="w_turn",
    )
    proteccion = st.number_input(
        "Protección ante rachas malas (más alto = evitar caídas)",
        value=float(rw.get("dd", 0.2)),
        help="Sube para evitar caídas; sugerido 0.2",
        key="w_dd",
    )
    suavidad = st.number_input(
        "Suavidad de resultados (más alto = menos diente de sierra)",
        value=float(rw.get("vol", 0.1)),
        help="Sube para suavizar; sugerido 0.1",
        key="w_vol",
    )

    stats = cfg.get("stats", {})
    env_caps = {"obs_type": "continuous", "action_type": "discrete", "state_space": stats.get("state_space", 100)}
    choice = choose_algo(stats, env_caps)
    algo = choice["algo"]
    cfg["algo"] = algo
    st.success(f"Algoritmo elegido: {algo} — {choice['reason']}")
    suggested = tune(algo, stats, [])
    if algo == "hybrid":
        ppo_sug = suggested.get("ppo", {})
        dqn_sug = suggested.get("dqn", {})
        st.subheader("Hiperparámetros críticos PPO")
        ppo_lr = st.number_input(
            "Velocidad de aprendizaje (qué tan rápido aprende)",
            value=float(ppo_sug.get("learning_rate", 3e-4)),
            format="%.6f",
            help=f"Sugerido {ppo_sug.get('learning_rate',3e-4):.2e}",
            key="ppo_lr",
        )
        ppo_batch = st.number_input(
            "Tamaño de lote (cada cuántos ejemplos actualiza)",
            value=int(ppo_sug.get("batch_size", 64)),
            help=f"Sugerido {ppo_sug.get('batch_size',64)}",
            key="ppo_batch",
        )
        ppo_steps = st.number_input(
            "Horizonte de actualización (pasos antes de actualizar)",
            value=int(ppo_sug.get("n_steps", 2048)),
            help=f"Sugerido {ppo_sug.get('n_steps',2048)}",
            key="ppo_steps",
        )
        st.subheader("Hiperparámetros críticos DQN")
        dqn_lr = st.number_input(
            "Velocidad de aprendizaje (qué tan rápido aprende) [DQN]",
            value=float(dqn_sug.get("learning_rate", 1e-3)),
            format="%.6f",
            help=f"Sugerido {dqn_sug.get('learning_rate',1e-3):.2e}",
            key="dqn_lr",
        )
        dqn_batch = st.number_input(
            "Tamaño de lote (cada cuántos ejemplos actualiza) [DQN]",
            value=int(dqn_sug.get("batch_size", 64)),
            help=f"Sugerido {dqn_sug.get('batch_size',64)}",
            key="dqn_batch",
        )
        dqn_steps = st.number_input(
            "Horizonte de actualización (pasos antes de actualizar) [DQN]",
            value=int(dqn_sug.get("n_steps", 1000)),
            help=f"Sugerido {dqn_sug.get('n_steps',1000)}",
            key="dqn_steps",
        )
        cfg["ppo"] = {
            "learning_rate": ppo_lr,
            "batch_size": int(ppo_batch),
            "n_steps": int(ppo_steps),
        }
        cfg["dqn"] = {
            "learning_rate": dqn_lr,
            "batch_size": int(dqn_batch),
            "target_update": int(dqn_steps),
        }
    else:
        lr = st.number_input(
            "Velocidad de aprendizaje (qué tan rápido aprende)",
            value=float(suggested.get("learning_rate", 3e-4)),
            format="%.6f",
            help=f"Sugerido {suggested.get('learning_rate',3e-4):.2e}",
        )
        batch = st.number_input(
            "Tamaño de lote (cada cuántos ejemplos actualiza)",
            value=int(suggested.get("batch_size", 64)),
            help=f"Sugerido {suggested.get('batch_size',64)}",
        )
        horizon = st.number_input(
            "Horizonte de actualización (pasos antes de actualizar)",
            value=int(suggested.get("n_steps", 2048)),
            help=f"Sugerido {suggested.get('n_steps',2048)}",
        )
        if algo == "ppo":
            cfg["ppo"] = {
                "learning_rate": lr,
                "batch_size": int(batch),
                "n_steps": int(horizon),
            }
        else:
            cfg["dqn"] = {
                "learning_rate": lr,
                "batch_size": int(batch),
                "target_update": int(horizon),
            }

    if st.button("💾 Guardar config YAML"):
        import yaml
        new_cfg = {
            "exchange": "binance",
            "binance_use_testnet": use_testnet,
            "symbols": selected_symbols,
            "timeframe": cfg.get("timeframe", "1m"),
            "fees": {"taker": fees_taker, "maker": cfg.get("fees", {}).get("maker", fees_taker)},
            "slippage_multiplier": slippage_mult,
            "min_notional_usd": min_notional,
            "filters": {"tickSize": tick_size, "stepSize": step_size},
            "algo": algo,
            "ppo": cfg.get("ppo", {}),
            "dqn": cfg.get("dqn", {}),
            "reward_weights": {
                "pnl": beneficio,
                "turn": control_act,
                "dd": proteccion,
                "vol": suavidad,
            },
            "paths": paths_cfg,
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)
        st.success(f"Guardado {CONFIG_PATH}")

st.subheader("📥 Datos")
st.caption("La precisión se elige automáticamente al mínimo disponible; el modelo puede reagrupar internamente")
st.write("Construyendo dataset con tramos de alta actividad...")
st.write("Seleccionados: " + ", ".join(selected_symbols))
if st.button("⬇️ Descargar histórico"):
    from datetime import datetime
    import pandas as pd
    try:
        tf_str = cfg.get("timeframe", "1m")
        timeframe_min = int(tf_str.rstrip("m"))
        st.info("Construyendo dataset con tramos de alta actividad...")
        windows = find_high_activity_windows(selected_symbols, timeframe_min)
        if windows:
            st.write("Ventanas ejemplo:")
            for s, e in windows[:5]:
                st.write(f"{datetime.utcfromtimestamp(s/1000)} → {datetime.utcfromtimestamp(e/1000)}")
        ex = get_exchange(use_testnet=use_testnet)
        for sym in selected_symbols:
            parts = []
            for s, e in windows:
                limit = int((e - s) / (timeframe_min * 60 * 1000))
                try:
                    df = fetch_ohlcv(ex, sym, timeframe=tf_str, since=s, limit=limit)
                    parts.append(df)
                except Exception as err:
                    st.warning(f"Fallo {sym}: {err}")
            if parts:
                merged = pd.concat(parts)
                tf = merged.attrs.get("timeframe", tf_str)
                cfg["timeframe"] = tf
                path = save_history(merged, str(raw_dir), "binance", sym, tf)
                st.success(f"Guardado: {path}")
        total_hours = sum((e - s) // 3600000 for s, e in windows)
        st.info(f"Ventanas total: {total_hours}h")
    except Exception as e:
        st.error(f"Error en descarga: {e}")

st.subheader("🧠 Entrenamiento")
colt1, colt2 = st.columns(2)
with colt1:
    st.caption(f"Algoritmo: {algo} — {choice['reason']}")
    timesteps = st.number_input("Timesteps", value=20000, step=1000)
with colt2:
    st.empty()
algo_run = algo

if st.button("🚀 Entrenar"):
    import tempfile, yaml
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
        cfg_path = tmp.name
    cmd = [
        "python",
        "-m",
        "src.training.train_drl",
        "--config",
        cfg_path,
        "--algo",
        algo_run,
        "--algo-reason",
        choice["reason"],
        "--timesteps",
        str(int(timesteps)),
    ]
    st.info("Ejecutando: " + " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.code(res.stdout or "", language="bash")
        if res.stderr:
            st.error(res.stderr)
    except Exception as e:
        st.error(f"Fallo al entrenar: {e}")

st.subheader("📊 Evaluación / Backtest")
colb1, colb2 = st.columns(2)
with colb1:
    policy = st.selectbox("Política", ["deterministic","stochastic","dqn"])
with colb2:
    st.empty()

if st.button("📈 Evaluar"):
    import tempfile, yaml
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
        cfg_path = tmp.name
    cmd = ["python", "-m", "src.backtest.evaluate", "--config", cfg_path, "--policy", policy]
    st.info("Ejecutando: " + " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.code(res.stdout or "", language="bash")
        if res.stderr:
            st.error(res.stderr)
    except Exception as e:
        st.error(f"Fallo al evaluar: {e}")

st.caption("Consejo: usa un terminal aparte si prefieres ver logs en tiempo real mientras el entrenamiento corre.")
