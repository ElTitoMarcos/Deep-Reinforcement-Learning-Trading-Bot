from dotenv import load_dotenv, find_dotenv
import os
_DOTENV = find_dotenv(usecwd=True)
load_dotenv(_DOTENV, override=True)
if __name__ == "__main__" or os.getenv("DEBUG_DOTENV") == "1":
    print(f"[.env] Cargado: {_DOTENV or 'NO ENCONTRADO'}")

import io, sys, json, subprocess, time, uuid, shutil
from datetime import datetime, UTC
import streamlit as st
from src.ui.log_stream import subscribe as log_subscribe, get_auto_profile, recent_counts
from pathlib import Path
from src.auto import reward_human_names, AlgoController
from src.ui.tasks import run_bg, poll, set_progress

from src.utils.config import load_config
from src.utils import paths
from src.reports.human_friendly import render_panel, to_human
from src.utils.device import get_device, set_cpu_threads
from src.data.ccxt_loader import get_exchange
from src.data.symbol_discovery import discover_symbols, discover_summary
from src.data.pipeline import prepare_data
from src.data import validate_symbols
from src.exchange.binance_meta import BinanceMeta
from src.auto.strategy_selector import choose_algo
from src.auto.hparam_tuner import tune
from src.utils.credentials import load_binance_creds, load_openai_key

CONFIG_PATH = st.session_state.get("config_path", "configs/default.yaml")

st.set_page_config(page_title="DRL Trading Config", layout="wide")

if "binance_mainnet_ok" not in st.session_state or "binance_testnet_ok" not in st.session_state:
    try:
        _, _, use_testnet = load_binance_creds()
        st.session_state["binance_testnet_ok"] = use_testnet
        st.session_state["binance_mainnet_ok"] = not use_testnet
    except Exception:
        st.session_state["binance_mainnet_ok"] = False
        st.session_state["binance_testnet_ok"] = False
        st.error("Faltan claves en .env; abre el panel de Conexiones para configurarlas")

if "openai_ok" not in st.session_state:
    try:
        load_openai_key()
        st.session_state["openai_ok"] = True
    except Exception:
        st.session_state["openai_ok"] = False
        st.error("Faltan claves en .env; abre el panel de Conexiones para configurarlas")

st.title("⚙️ Configuración DRL Trading")

if "fee_taker" not in st.session_state:
    st.session_state["fee_taker"] = None
if "fee_maker" not in st.session_state:
    st.session_state["fee_maker"] = None
if "busy" not in st.session_state:
    st.session_state["busy"] = False

device = get_device()
if device == "cuda":
    import torch

    name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"Dispositivo: CUDA ({name})")
else:
    threads = set_cpu_threads()
    st.sidebar.info(f"Dispositivo: CPU ({threads} hilos)")
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True)
            if res.returncode == 0:
                st.sidebar.warning(
                    "Se detectó GPU pero PyTorch no tiene soporte CUDA; considera reinstalarlo con CUDA."
                )
        except Exception:
            pass

with st.sidebar:
    st.header("Conexiones")
    def badge(ok: bool) -> str:
        return f":{'green' if ok else 'red'}[{'Conectado' if ok else 'No conectado'}]"

    st.markdown(f"Binance mainnet {badge(st.session_state.get('binance_mainnet_ok', False))}")
    st.markdown(f"Binance testnet {badge(st.session_state.get('binance_testnet_ok', False))}")
    st.markdown(f"OpenAI {badge(st.session_state.get('openai_ok', False))}")
    if not (
        st.session_state.get("binance_mainnet_ok")
        and st.session_state.get("binance_testnet_ok")
        and st.session_state.get("openai_ok")
    ):
        st.warning("Faltan claves o verificación falló")
        try:
            st.page_link("src/ui/credentials.py", label="Configurar conexiones")
        except Exception:
            st.markdown("[Configurar conexiones](./credentials)")

    st.header("Ajustes globales")
    CONFIG_PATH = st.text_input("Ruta config YAML", value=CONFIG_PATH, key="cfg_path", help="Normalmente configs/default.yaml")
    st.session_state["config_path"] = CONFIG_PATH
    # Cargar YAML
    try:
        cfg = load_config(CONFIG_PATH)
        defaults_used = cfg.pop("_defaults_used", [])
        paths.ensure_dirs_exist()
        if defaults_used:
            st.warning("Config incompleta; se aplicaron defaults para: " + ", ".join(defaults_used))
    except Exception as e:
        st.error(f"No se pudo cargar {CONFIG_PATH}: {e}")
        cfg = {}

    paths_cfg = cfg.get("paths", {})
    raw_dir = paths.RAW_DIR
    use_testnet_default = bool(cfg.get("binance_use_testnet", False))
    mode = st.radio("Modo", ["Mainnet", "Testnet"], index=1 if use_testnet_default else 0)
    use_testnet = mode == "Testnet"
    os.environ["BINANCE_USE_TESTNET"] = "true" if use_testnet else "false"
    try:
        ex = get_exchange(use_testnet=use_testnet)
        selected_symbols = discover_symbols(ex, top_n=20)
    except RuntimeError:
        st.error("Faltan claves en .env; abre el panel de Conexiones para configurarlas")
        selected_symbols = cfg.get("symbols") or ["BTC/USDT"]
    except Exception as e:
        st.warning(f"Descubrimiento falló: {e}")
        selected_symbols = cfg.get("symbols") or ["BTC/USDT"]
    cfg["symbols"] = selected_symbols
    st.caption(discover_summary(selected_symbols))
    st.code("\n".join(selected_symbols))

    fees_dict = cfg.get("fees", {})
    default_fee_taker = float(fees_dict.get("taker", 0.001))
    api_fee_taker = None
    api_fee_maker = None

    header_col, badge_col = st.columns([3, 2])
    with header_col:
        st.subheader("Comisiones")
    with badge_col:
        origin = st.session_state.get("fee_origin")
        if origin:
            st.caption(origin)

    if st.button("Actualizar comisiones"):
        try:
            key, sec, _ = load_binance_creds()
            meta = BinanceMeta(key, sec, use_testnet)
            fee_map = meta.get_account_trade_fees()
            if "maker" in fee_map and "taker" in fee_map:
                entry = fee_map
            else:
                symbol_key = (
                    selected_symbols[0].replace("/", "") if selected_symbols else next(iter(fee_map))
                )
                entry = fee_map.get(symbol_key) or next(iter(fee_map.values()))
            api_fee_taker = entry.get("taker")
            api_fee_maker = entry.get("maker")
            st.session_state["fee_origin"] = meta.last_fee_origin
            st.success(f"Maker {api_fee_maker} | Taker {api_fee_taker}")
        except RuntimeError:
            st.error("Faltan claves en .env; abre el panel de Conexiones para configurarlas")
        except Exception as e:
            st.error(f"No se pudo obtener: {e}")

    fee_taker = api_fee_taker or st.session_state["fee_taker"] or default_fee_taker
    fee_maker = api_fee_maker or st.session_state["fee_maker"] or fee_taker
    st.session_state["fee_taker"] = fee_taker
    st.session_state["fee_maker"] = fee_maker

    st.number_input(
        "Fee taker",
        value=float(fee_taker),
        step=0.0001,
        format="%.6f",
        key="fee_taker",
    )
    st.number_input(
        "Fee maker",
        value=float(fee_maker),
        step=0.0001,
        format="%.6f",
        key="fee_maker",
    )
    cfg["fees"] = {
        "maker": st.session_state["fee_maker"] or fee_maker,
        "taker": st.session_state["fee_taker"] or fee_taker,
    }
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
    tab_manual, tab_auto, tab_strategy = st.tabs(["Manual", "Auto-tuning", "Estrategia"])
    with tab_manual:
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
    cfg["reward_weights"] = {
        "pnl": beneficio,
        "turn": control_act,
        "dd": proteccion,
        "vol": suavidad,
    }

    with tab_auto:
        import pandas as pd

        rt_cfg = cfg.get("reward_tuner", {})
        enabled = st.checkbox(
            "Ajustar pesos automáticamente",
            value=bool(rt_cfg.get("enabled", True)),
        )
        freq = st.number_input(
            "Frecuencia (episodios)",
            value=int(rt_cfg.get("freq_episodes", 50)),
            min_value=1,
            step=1,
            help="Cada cuántos episodios proponer ajustes",
        )
        delta = st.number_input(
            "Amplitud máxima por iteración",
            value=float(rt_cfg.get("delta", 0.05)),
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )
        cfg["reward_tuner"] = {
            **rt_cfg,
            "enabled": bool(enabled),
            "freq_episodes": int(freq),
            "delta": float(delta),
        }

        hist_file = Path("reports/reward_tuning_history.jsonl")
        weights = {
            "w_pnl": st.session_state.get("w_pnl", rw.get("pnl", 1.0)),
            "w_drawdown": st.session_state.get("w_dd", rw.get("dd", 0.2)),
            "w_volatility": st.session_state.get("w_vol", rw.get("vol", 0.1)),
            "w_turnover": st.session_state.get("w_turn", rw.get("turn", 0.1)),
        }
        diffs = {k: 0.0 for k in weights}
        scores: list[float] = []
        if hist_file.exists():
            lines = hist_file.read_text().splitlines()
            if lines:
                last = json.loads(lines[-1])
                weights = last.get("after", weights)
                before = last.get("before", {})
                for k in weights:
                    diffs[k] = weights[k] - before.get(k, weights[k])
                scores = [json.loads(l).get("score_after", 0.0) for l in lines[-20:]]
        names = reward_human_names()
        rows = []
        for k in ["w_pnl", "w_drawdown", "w_volatility", "w_turnover"]:
            val = weights.get(k, 0.0)
            diff = diffs.get(k, 0.0)
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            color = "green" if diff > 0 else "red" if diff < 0 else "gray"
            rows.append(
                {
                    "Peso": names.get(k, k),
                    "Valor": f"{val:.2f}",
                    "Cambio": f"<span style='color:{color}'>{arrow} {diff:+.2f}</span>",
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        if scores:
            st.line_chart(scores)

    with tab_strategy:
        ctrl: AlgoController = st.session_state.get("algo_ctrl") or AlgoController()
        fixed = st.session_state.get("algo_fixed", False)
        if st.button("Desfijar" if fixed else "Fijar", key="fix_algo"):
            fixed = not fixed
        ctrl.fixed = fixed
        mapping = ctrl.decide({}, {}, {})
        st.json(mapping)
        st.caption(ctrl.explain(mapping))
        st.session_state["algo_ctrl"] = ctrl
        st.session_state["algo_fixed"] = fixed

    stats = cfg.get("stats", {})
    env_caps = {
        "obs_type": "continuous",
        "action_type": "discrete",
        "state_space": stats.get("state_space", 100),
    }
    choice = choose_algo(stats, env_caps)
    algo = choice["algo"]
    cfg["algo"] = algo
    suggested = tune(algo, stats, [])
    if algo == "hybrid":
        ppo_sug = suggested.get("ppo", {})
        dqn_sug = suggested.get("dqn", {})

        def _avg(key: str, default: float | int) -> float:
            p = ppo_sug.get(key)
            d = dqn_sug.get("target_update" if key == "n_steps" else key)
            vals = [v for v in [p, d] if v is not None]
            return float(sum(vals) / len(vals)) if vals else float(default)

        suggested_flat = {
            "learning_rate": _avg("learning_rate", 3e-4),
            "batch_size": int(_avg("batch_size", 64)),
            "n_steps": int(_avg("n_steps", 2048)),
        }
    else:
        suggested_flat = {
            "learning_rate": float(suggested.get("learning_rate", 3e-4)),
            "batch_size": int(suggested.get("batch_size", 64)),
            "n_steps": int(suggested.get("n_steps", suggested.get("target_update", 2048))),
        }

    lr = st.number_input(
        "Velocidad de aprendizaje",
        value=float(suggested_flat["learning_rate"]),
        format="%.6f",
        help=f"Sugerido {suggested_flat['learning_rate']:.2e}",
    )
    batch = st.number_input(
        "Tamaño de lote",
        value=int(suggested_flat["batch_size"]),
        help=f"Sugerido {suggested_flat['batch_size']}",
    )
    horizon = st.number_input(
        "Horizonte de actualización",
        value=int(suggested_flat["n_steps"]),
        help=f"Sugerido {suggested_flat['n_steps']}",
    )
    selected = {"learning_rate": lr, "batch_size": int(batch), "n_steps": int(horizon)}

    if algo == "ppo":
        cfg["ppo"] = selected
    elif algo == "dqn":
        cfg["dqn"] = {
            "learning_rate": lr,
            "batch_size": int(batch),
            "target_update": int(horizon),
        }
    else:  # hybrid
        cfg["ppo"] = selected
        cfg["dqn"] = {
            "learning_rate": lr,
            "batch_size": int(batch),
            "target_update": int(horizon),
        }

    if st.button("Explicación"):
        st.info(f"Algoritmo elegido: {algo}")
        st.write(choice["reason"])
        diffs = {
            k: (suggested_flat[k], selected[k])
            for k in selected
            if selected[k] != suggested_flat[k]
        }
        if diffs:
            st.write("Ajustes modificados:")
            for k, (sug, val) in diffs.items():
                st.write(f"{k}: {sug} -> {val}")
        else:
            st.write("Se usan hiperparámetros sugeridos sin cambios.")

    st.header("Asistente LLM")
    llm_model = st.selectbox(
        "Modelo",
        ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
        index=0,
    )
    llm_reason = st.checkbox("Usar LLM para decisiones razonadas")
    llm_periodic = st.checkbox("Llamadas periódicas durante entrenamiento")
    llm_mode = "Por episodios"
    llm_every = 0
    if llm_periodic:
        llm_mode = st.radio("Modo", ["Por episodios", "Por minutos"], index=0)
        if llm_mode == "Por episodios":
            llm_every = st.number_input(
                "Frecuencia de consultas al asistente (en episodios)",
                value=0,
                min_value=0,
                step=1,
                help="Ej.: 50 = pedirá consejo al asistente cada 50 episodios. 0 = desactivado.",
            )
        else:
            llm_every = st.number_input(
                "Frecuencia de consultas al asistente (en minutos)",
                value=0,
                min_value=0,
                step=1,
                help="Ej.: 5 = pedirá consejo al asistente cada 5 minutos. 0 = desactivado.",
            )
    periodic_enabled = bool(llm_periodic and llm_every > 0)
    cfg["llm"] = {
        "model": llm_model,
        "enabled": bool(llm_reason or periodic_enabled),
        "use_reasoned": bool(llm_reason),
        "periodic": periodic_enabled,
        "mode": "minutes" if llm_mode == "Por minutos" else "episodes",
        "every_n": int(llm_every),
    }

    if st.button("💾 Guardar config YAML"):
        import yaml
        new_cfg = {
            "exchange": "binance",
            "binance_use_testnet": use_testnet,
            "symbols": selected_symbols,
            "timeframe": cfg.get("timeframe", "1m"),
            "fees": {"taker": fee_taker, "maker": fee_maker},
            "slippage_multiplier": slippage_mult,
            "min_notional_usd": min_notional,
            "filters": {"tickSize": tick_size, "stepSize": step_size},
            "algo": algo,
            "ppo": cfg.get("ppo", {}),
            "dqn": cfg.get("dqn", {}),
            "llm": cfg.get("llm", {}),
            "reward_weights": {
                "pnl": beneficio,
                "turn": control_act,
                "dd": proteccion,
                "vol": suavidad,
            },
            "reward_tuner": cfg.get("reward_tuner", {}),
            "paths": paths_cfg,
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)
        st.success(f"Guardado {CONFIG_PATH}")

try:
    ex_val = get_exchange(use_testnet=use_testnet)
    selected_valid, invalid_syms = validate_symbols(ex_val, selected_symbols)
except RuntimeError:
    st.error("Faltan claves en .env; abre el panel de Conexiones para configurarlas")
    selected_valid, invalid_syms = selected_symbols, []
except Exception as e:
    st.warning(f"No se pudo validar símbolos: {e}")
    selected_valid, invalid_syms = selected_symbols, []
if invalid_syms:
    st.error("Símbolos inválidos")
    for item in invalid_syms:
        msg = item["reason"]
        if item.get("suggest"):
            msg += f"; quizá quisiste decir {item['suggest']}?"
        st.write(f"{item['symbol']}: {msg}")
selected_symbols = selected_valid
cfg["symbols"] = selected_valid

st.subheader("📥 Datos")
if "prep_job" not in st.session_state:
    st.session_state["prep_job"] = None

status_box = st.empty()

job_id = st.session_state.get("prep_job")
if job_id:
    info = poll(job_id)
    label = info.get("progress") or "Preparando…"
    state = info.get("state")
    if state == "running":
        status_box.status(label, expanded=True)
        time.sleep(0.5)
        st.rerun()
    elif state == "done":
        status_box.status(label or "Refresco en marcha ✔", state="complete")
        st.session_state["data_ready"] = True
        st.session_state["prep_job"] = None
        st.rerun()
    elif state == "error":
        status_box.status(f"Error: {info.get('error')}", state="error")
        st.session_state["prep_job"] = None
else:
    if st.button("Preparar datos (auto)"):
        job_id = f"prepare_data-{uuid.uuid4().hex[:8]}"

        def _report(msg: str, j=job_id) -> None:
            set_progress(j, msg)

        run_bg("prepare_data", prepare_data, job_id=job_id, progress_cb=_report)
        st.session_state["prep_job"] = job_id
        st.rerun()

st.subheader("🧠 Entrenamiento")
colt1, colt2 = st.columns(2)
with colt1:
    st.caption(f"Algoritmo: {algo} — {choice['reason']}")
    st.caption(
        "Los timesteps se adaptan automáticamente por etapa (puedes activar asesor LLM en Ajustes)."
    )
with colt2:
    st.empty()
algo_run = algo

if st.button("🚀 Entrenar"):
    import tempfile, yaml, threading, sys

    st.session_state["busy"] = True
    try:
        if invalid_syms:
            st.warning(
                "Ignorando símbolos inválidos: "
                + ", ".join(i["symbol"] for i in invalid_syms)
            )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
            cfg_path = tmp.name

        def _run_train():
            from src.training import train_drl

            sys.argv = [
                "train_drl",
                "--config",
                cfg_path,
                "--algo",
                algo_run,
                "--algo-reason",
                choice["reason"],
                "--timesteps",
                str(int(1_000_000)),
            ]
            try:
                train_drl.main()
            except Exception as e:  # pragma: no cover - user feedback
                st.error(f"Fallo al entrenar: {e}")

        log_box = st.empty()
        progress_bar = st.progress(0.0)
        progress_stats = st.empty()
        target_steps = int(1_000_000)
        stage = "training"
        thread = threading.Thread(target=_run_train, daemon=True)
        thread.start()
        lines: list[str] = []
        log_iter = log_subscribe(level="info")
        while thread.is_alive():
            try:
                entry = next(log_iter)
                msg = entry["message"]
                lines.append(f"[{entry['kind']}] {msg}")
                log_box.code("\n".join(lines[-200:]))
                if msg.startswith("[heartbeat]"):
                    parts = dict(p.split("=") for p in msg.replace("[heartbeat]", "").strip().split())
                    steps = int(parts.get("steps", 0))
                    reward = float(parts.get("reward_mean", 0.0))
                    progress_bar.progress(min(steps / target_steps, 1.0))
                    progress_stats.text(
                        f"Etapa: {stage} | Pasos: {steps}/{target_steps} | Reward medio: {reward:.4f}"
                    )
            except Exception:
                pass
        thread.join()
        # Drain any remaining log lines
        for _ in range(50):
            try:
                entry = next(log_iter)
                msg = entry["message"]
                lines.append(f"[{entry['kind']}] {msg}")
                if msg.startswith("[heartbeat]"):
                    parts = dict(p.split("=") for p in msg.replace("[heartbeat]", "").strip().split())
                    steps = int(parts.get("steps", 0))
                    reward = float(parts.get("reward_mean", 0.0))
                    progress_bar.progress(min(steps / target_steps, 1.0))
                    progress_stats.text(
                        f"Etapa: {stage} | Pasos: {steps}/{target_steps} | Reward medio: {reward:.4f}"
                    )
            except Exception:
                break
        log_box.code("\n".join(lines[-200:]))
        st.success("Entrenamiento finalizado")
    except BaseException as err:
        if isinstance(err, Exception):
            st.error(f"Error: {err}")
        else:
            st.warning("Proceso cancelado")
    finally:
        st.session_state["busy"] = False

st.subheader("📊 Evaluación / Backtest")
st.caption("La política se elige automáticamente según el algoritmo entrenado")

if st.button("📈 Evaluar"):
    import tempfile, yaml
    st.session_state["busy"] = True
    try:
        if invalid_syms:
            st.warning(
                "Ignorando símbolos inválidos: "
                + ", ".join(i["symbol"] for i in invalid_syms)
            )
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
            cfg_path = tmp.name
        cmd = ["python", "-m", "src.backtest.evaluate", "--config", cfg_path]
        st.info("Ejecutando: " + " ".join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            logs = res.stdout or ""
            if logs:
                st.expander("Logs").code(logs, language="bash")

            reports_root = paths.reports_dir()
            run_dirs = sorted(
                reports_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if run_dirs:
                latest = run_dirs[0]
                try:
                    with open(latest / "metrics.json") as f:
                        results = json.load(f)
                    render_panel(results)
                    st.caption(f"Resumen guardado en {latest}")
                except Exception as err:
                    st.error(f"No se pudo leer métricas: {err}")
            else:
                st.warning("No hay reportes disponibles")

            if res.stderr:
                st.error(res.stderr)
        except Exception as e:
            st.error(f"Fallo al evaluar: {e}")
    except BaseException as err:
        if isinstance(err, Exception):
            st.error(f"Error: {err}")
        else:
            st.warning("Proceso cancelado")
    finally:
        st.session_state["busy"] = False

# ---- Registro por áreas ---------------------------------------------------

st.subheader("Registro")
total_events, kind_counts = recent_counts()
parts = ", ".join(f"{k} {v}" for k, v in kind_counts.items())
st.caption(f"Últimos 30s: {total_events} eventos ({parts})")

AREA_KINDS = {
    "Datos": {"datos", "incremental_update", "qc"},
    "Entrenamiento": {"reward_tuner", "dqn_stability", "checkpoints"},
    "Evaluación": {"hybrid_weights", "performance"},
    "LLM": {"llm"},
    "Riesgo": {"riesgo"},
}

tabs = st.tabs(list(AREA_KINDS.keys()))

if "log_iters" not in st.session_state:
    st.session_state["log_iters"] = {}
if "log_buffers" not in st.session_state:
    st.session_state["log_buffers"] = {}

for tab, (area, kinds) in zip(tabs, AREA_KINDS.items()):
    if area not in st.session_state["log_iters"]:
        st.session_state["log_iters"][area] = log_subscribe(kinds=kinds)
        st.session_state["log_buffers"][area] = []
    with tab:
        placeholder = st.empty()
        if not st.session_state.get("busy"):
            start = time.time()
            gen = st.session_state["log_iters"][area]
            buf = st.session_state["log_buffers"][area]
            while time.time() - start < 0.5:
                try:
                    item = next(gen)
                    buf.append(item)
                except StopIteration:
                    break
                except Exception:
                    break
            st.session_state["log_buffers"][area] = buf[-200:]
        lines = [to_human(it) for it in st.session_state["log_buffers"][area]]
        placeholder.text("\n".join(lines))
        if st.button("Exportar", key=f"export_{area}"):
            run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            run_dir = paths.reports_dir() / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "log_humano.md"
            md_lines = [
                f"- {it['time'].isoformat()} - {to_human(it)}"
                for it in st.session_state["log_buffers"][area][-200:]
            ]
            path.write_text("\n".join(md_lines), encoding="utf-8")
            st.success(f"Guardado en {path}")

if not st.session_state.get("busy"):
    time.sleep(0.5)
    st.rerun()
