import os, io, sys, json, subprocess, time
from datetime import datetime
import streamlit as st

from src.utils.config import load_config
from src.utils.paths import ensure_dirs_exist, get_raw_dir
from src.data.ccxt_loader import get_exchange, fetch_ohlcv, simulate_1s_from_1m, save_history

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

    symbols = st.text_input("Símbolos (espacio-separado)", value=" ".join(cfg.get("symbols") or ["BTC/USDT"]), key="symbols_space")
    timeframe = st.selectbox("Timeframe", ["1s","1m","3m","5m","15m"], index=1)

    fees_taker = st.number_input("Fee taker", value=float(cfg.get("fees",{}).get("taker",0.001)), step=0.0001, format="%.6f")
    slippage = st.number_input("Slippage", value=float(cfg.get("slippage",0.0005)), step=0.0001, format="%.6f")
    min_notional = st.number_input("Mínimo notional USD", value=float(cfg.get("min_notional_usd",10.0)), step=1.0)

    tick_size = st.number_input("tickSize", value=float(cfg.get("filters",{}).get("tickSize",0.01)))
    step_size = st.number_input("stepSize", value=float(cfg.get("filters",{}).get("stepSize",0.0001)))

    st.caption("Reward heads (pesos)")
    rw = cfg.get("reward_weights", {"pnl":1.0,"turnover_penalty":0.1,"drawdown_penalty":0.2,"volatility_penalty":0.1})
    w_pnl = st.number_input("w_pnl", value=float(rw.get("pnl",1.0)))
    w_turn = st.number_input("w_turnover", value=float(rw.get("turnover_penalty",0.1)))
    w_dd = st.number_input("w_drawdown", value=float(rw.get("drawdown_penalty",0.2)))
    w_vol = st.number_input("w_volatility", value=float(rw.get("volatility_penalty",0.1)))

    st.caption("Algoritmo")
    algo = st.selectbox("Algo", ["ppo", "dqn"], index=0 if (cfg.get("algo","ppo")=="ppo") else 1)
    ppo = cfg.get("ppo", {})
    dqn = cfg.get("dqn", {})

    with st.expander("Hiperparámetros PPO"):
        ppo_lr = st.number_input("learning_rate", value=float(ppo.get("learning_rate",3e-4)), format="%.8f")
        ppo_steps = st.number_input("n_steps", value=int(ppo.get("n_steps",2048)))
        ppo_batch = st.number_input("batch_size", value=int(ppo.get("batch_size",64)))
        ppo_gamma = st.number_input("gamma", value=float(ppo.get("gamma",0.99)))
        ppo_lambda = st.number_input("gae_lambda", value=float(ppo.get("gae_lambda",0.95)))
        ppo_clip = st.number_input("clip_range", value=float(ppo.get("clip_range",0.2)))
        ppo_ent = st.number_input("ent_coef", value=float(ppo.get("ent_coef",0.01)))

    with st.expander("Hiperparámetros DQN"):
        dqn_lr = st.number_input("learning_rate ", value=float(dqn.get("learning_rate",1e-3)), format="%.8f")
        dqn_gamma = st.number_input("gamma ", value=float(dqn.get("gamma",0.99)))
        dqn_batch = st.number_input("batch_size ", value=int(dqn.get("batch_size",64)))
        dqn_target = st.number_input("target_update ", value=int(dqn.get("target_update",1000)))
        dqn_eps_s = st.number_input("epsilon_start", value=float(dqn.get("epsilon_start",1.0)))
        dqn_eps_e = st.number_input("epsilon_end", value=float(dqn.get("epsilon_end",0.05)))
        dqn_eps_d = st.number_input("epsilon_decay_steps", value=int(dqn.get("epsilon_decay_steps",10000)))

    if st.button("💾 Guardar config YAML"):
        import yaml
        new_cfg = {
            "exchange": "binance",
            "binance_use_testnet": use_testnet,
            "symbols": symbols.split(),
            "timeframe": timeframe,
            "fees": {"taker": fees_taker, "maker": cfg.get("fees", {}).get("maker", fees_taker)},
            "slippage": slippage,
            "min_notional_usd": min_notional,
            "filters": {"tickSize": tick_size, "stepSize": step_size},
            "algo": algo,
            "ppo": {
                "learning_rate": ppo_lr, "n_steps": int(ppo_steps), "batch_size": int(ppo_batch),
                "gamma": ppo_gamma, "gae_lambda": ppo_lambda, "clip_range": ppo_clip, "ent_coef": ppo_ent
            },
            "dqn": {
                "learning_rate": dqn_lr, "gamma": dqn_gamma, "batch_size": int(dqn_batch),
                "target_update": int(dqn_target), "epsilon_start": dqn_eps_s,
                "epsilon_end": dqn_eps_e, "epsilon_decay_steps": int(dqn_eps_d)
            },
            "reward_weights": {"pnl": w_pnl, "turnover_penalty": w_turn, "drawdown_penalty": w_dd, "volatility_penalty": w_vol},
            "paths": paths_cfg,
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)
        st.success(f"Guardado {CONFIG_PATH}")

st.subheader("📥 Datos")
col1, col2 = st.columns(2)
with col1:
    dl_timeframe = st.selectbox("Timeframe descarga", ["1s","1m","3m","5m","15m"], index=1, key="dl_tf")
with col2:
    since = st.text_input("Desde (ISO UTC, ej. 2024-01-01)", value="", key="since_iso")

symbols_run = st.text_input("Símbolos a descargar", value=symbols, key="symbols_dl")
if st.button("⬇️ Descargar histórico"):
    try:
        ex = get_exchange(use_testnet=use_testnet)
        from datetime import datetime, timezone
        since_ms = None
        if since:
            try:
                dt = datetime.fromisoformat(since.replace("Z","")).replace(tzinfo=timezone.utc)
                since_ms = int(dt.timestamp()*1000)
            except Exception:
                since_ms = None
        for sym in symbols_run.split():
            df = fetch_ohlcv(ex, sym, timeframe=dl_timeframe, since=since_ms)
            if dl_timeframe == "1s" and df.empty:
                st.warning(f"1s no disponible en {sym}; simulando desde 1m")
                df_1m = fetch_ohlcv(ex, sym, timeframe="1m", since=since_ms)
                df = simulate_1s_from_1m(df_1m)
            path = save_history(df, str(raw_dir), "binance", sym, dl_timeframe)
            st.success(f"Guardado: {path}")
    except Exception as e:
        st.error(f"Error en descarga: {e}")

st.subheader("🧠 Entrenamiento")
colt1, colt2 = st.columns(2)
with colt1:
    algo_run = st.selectbox("Algoritmo", ["ppo","dqn"], index=0 if algo=="ppo" else 1)
    timesteps = st.number_input("Timesteps", value=20000, step=1000)
with colt2:
    st.empty()

if st.button("🚀 Entrenar"):
    cmd = ["python", "-m", "src.training.train_drl", "--config", CONFIG_PATH, "--algo", algo_run, "--timesteps", str(int(timesteps))]
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
    cmd = ["python", "-m", "src.backtest.evaluate", "--config", CONFIG_PATH, "--policy", policy]
    st.info("Ejecutando: " + " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.code(res.stdout or "", language="bash")
        if res.stderr:
            st.error(res.stderr)
    except Exception as e:
        st.error(f"Fallo al evaluar: {e}")

st.caption("Consejo: usa un terminal aparte si prefieres ver logs en tiempo real mientras el entrenamiento corre.")
