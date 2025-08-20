import os, io, sys, json, subprocess, time
from datetime import datetime, UTC
import streamlit as st
from src.ui.log_stream import subscribe as log_subscribe
from pathlib import Path
from src.auto import reward_human_names, AlgoController

from src.utils.config import load_config
from src.utils import paths
from src.reports.human_friendly import render_panel
from src.utils.device import get_device, set_cpu_threads
from src.data.ccxt_loader import get_exchange, save_history
from src.data.ensure import ensure_ohlcv
from src.data.volatility_windows import find_high_activity_windows
from src.data.symbol_discovery import discover_symbols
from src.data import (
    fetch_symbol_metadata,
    fetch_extra_series,
    validate_symbols,
    validate_ohlcv,
    validate_metadata,
    validate_trades,
    passes,
    summarize,
)
from src.data.quality import QualityReport
from src.exchange.binance_meta import BinanceMeta
from dotenv import load_dotenv
from src.auto.strategy_selector import choose_algo
from src.auto.hparam_tuner import tune

CONFIG_PATH = st.session_state.get("config_path", "configs/default.yaml")

st.set_page_config(page_title="DRL Trading Config", layout="wide")

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

with st.sidebar:
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
    default_fee_taker = float(fees_dict.get("taker", 0.001))
    api_fee_taker = None
    api_fee_maker = None
    btn_col, origin_col = st.columns([1, 1])
    with btn_col:
        if st.button("Actualizar comisiones"):
            load_dotenv()
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            try:
                meta = BinanceMeta(api_key, api_secret, use_testnet)
                fee_map = meta.get_account_trade_fees()
                symbol_key = (
                    selected_symbols[0].replace("/", "") if selected_symbols else next(iter(fee_map))
                )
                entry = fee_map.get(symbol_key) or next(iter(fee_map.values()))
                api_fee_taker = entry.get("taker")
                api_fee_maker = entry.get("maker")
                st.session_state["fee_origin"] = meta.last_fee_origin
                st.success(f"Maker {api_fee_maker} | Taker {api_fee_taker}")
            except Exception as e:
                st.error(f"No se pudo obtener: {e}")
    with origin_col:
        origin = st.session_state.get("fee_origin")
        if origin:
            st.caption(origin)

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

st.subheader("🧹 Enriquecimiento y verificación de datos")
st.caption(
    "Descarga datos iniciales y los valida. Usa '🔄 Actualizar datos' para traer solo nuevos registros."
)
if st.button("Obtener y validar datos"):
    from pathlib import Path
    from datetime import datetime

    st.session_state["busy"] = True
    try:
        with st.spinner("Descargando y validando..."):
            ex = get_exchange(use_testnet=use_testnet)
            # Re-descubrir por si hay nuevos símbolos disponibles
            try:
                discover_symbols(ex, top_n=5)
            except Exception:
                pass
            if invalid_syms:
                st.warning(
                    "Ignorando símbolos inválidos: "
                    + ", ".join(i["symbol"] for i in invalid_syms)
                )

            meta_map = fetch_symbol_metadata(selected_symbols)
            for sym in selected_symbols:
                meta = meta_map.get(sym, {})
                m_report = validate_metadata(meta)
                series = fetch_extra_series(sym, timeframe=cfg.get("timeframe", "1m"))
                ohlcv = series.get("ohlcv")
                t_report = validate_trades(series.get("trades"))
                o_report = validate_ohlcv(ohlcv)
                combined = QualityReport()
                combined.errors.extend(m_report.errors + o_report.errors + t_report.errors)
                combined.warnings.extend(
                    m_report.warnings + o_report.warnings + t_report.warnings
                )
                summary = summarize(combined)
                if passes(combined):
                    out_dir = Path("data/processed") / sym.replace("/", "")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    data_file = ""
                    if ohlcv is not None and not ohlcv.empty:
                        try:
                            ohlcv.reset_index().to_parquet(
                                out_dir / "ohlcv.parquet", index=False
                            )
                            data_file = "ohlcv.parquet"
                        except Exception:
                            ohlcv.reset_index().to_csv(
                                out_dir / "ohlcv.csv", index=False
                            )
                            data_file = "ohlcv.csv"
                    manifest = {
                        "symbol": sym,
                        "obtained_at": datetime.now(UTC).isoformat(),
                        "source": meta.get("source"),
                        "qc": summary,
                        "data_file": data_file,
                    }
                    if meta.get("error"):
                        manifest["note"] = meta["error"]
                    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2)
                    st.success(f"✅ {sym} - {summary}")
                else:
                    st.error(f"❌ {sym} - {summary}")
        st.success("Proceso completado")
    except BaseException as err:
        if isinstance(err, Exception):
            st.error(f"Error: {err}")
        else:
            st.warning("Proceso cancelado")
    finally:
        st.session_state["busy"] = False

st.subheader("📥 Datos")
st.caption("La precisión se elige automáticamente al mínimo disponible; el modelo puede reagrupar internamente")
st.write("Construyendo dataset con tramos de alta actividad...")
st.write("Seleccionados: " + ", ".join(selected_symbols))
if st.button("🔄 Actualizar datos"):
    from datetime import datetime, UTC, timedelta
    import json
    from src.data.incremental import (
        last_watermark,
        fetch_ohlcv_incremental,
        upsert_parquet,
    )

    st.session_state["busy"] = True
    try:
        ex = get_exchange(use_testnet=use_testnet)
        tf_str = cfg.get("timeframe", "1m")
        for sym in selected_symbols:
            since = last_watermark(sym, tf_str)
            if since is None:
                since = int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000)
            df_new = fetch_ohlcv_incremental(ex, sym, tf_str, since_ms=since)
            if df_new.empty:
                st.info(f"{sym}: sin datos nuevos")
                continue
            path = paths.raw_parquet_path(ex.id if hasattr(ex, "id") else "binance", sym, tf_str)
            upsert_parquet(df_new, path)
            manifest = {
                "symbol": sym,
                "timeframe": tf_str,
                "watermark": int(df_new["ts"].max()),
                "obtained_at": datetime.now(UTC).isoformat(),
            }
            with open(path.with_suffix(".manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            st.success(f"{sym} actualizado")
    except BaseException as err:
        if isinstance(err, Exception):
            st.error(f"Error: {err}")
        else:
            st.warning("Proceso cancelado")
    finally:
        st.session_state["busy"] = False

if st.button("⬇️ Descargar histórico"):
    from datetime import datetime
    import pandas as pd
    st.session_state["busy"] = True
    try:
        if invalid_syms:
            st.warning(
                "Ignorando símbolos inválidos: "
                + ", ".join(i["symbol"] for i in invalid_syms)
            )
        tf_str = cfg.get("timeframe", "1m")
        timeframe_min = int(tf_str.rstrip("m"))
        st.info("Construyendo dataset con tramos de alta actividad...")
        windows, lookback_h = find_high_activity_windows(
            selected_symbols, timeframe_min
        )
        if windows:
            st.write("Ventanas ejemplo:")
            for s, e in windows[:5]:
                st.write(
                    f"{datetime.fromtimestamp(s/1000, UTC)} → {datetime.fromtimestamp(e/1000, UTC)}"
                )
        total_hours = sum((e - s) // 3600000 for s, e in windows)
        st.info(f"Ventanas total: {total_hours}h")
        ex = get_exchange(use_testnet=use_testnet)
        for sym in selected_symbols:
            try:
                path = ensure_ohlcv(
                    ex.id if hasattr(ex, "id") else "binance",
                    sym,
                    tf_str,
                    hours=lookback_h,
                )
                df = pd.read_parquet(path)
                parts = [df[(df.ts >= s) & (df.ts < e)] for s, e in windows]
                if parts:
                    merged = pd.concat(parts)
                    tf = tf_str
                    cfg["timeframe"] = tf
                    out = save_history(
                        merged,
                        paths.RAW_DIR,
                        ex.id if hasattr(ex, "id") else "binance",
                        sym,
                        tf,
                    )
                    st.success(f"Guardado: {out}")
            except Exception as err:
                st.warning(f"Fallo {sym}: {err}")
    except BaseException as err:
        if isinstance(err, Exception):
            st.error(f"Error en descarga: {err}")
        else:
            st.warning("Proceso cancelado")
    finally:
        st.session_state["busy"] = False

st.subheader("🧠 Entrenamiento")
colt1, colt2 = st.columns(2)
with colt1:
    st.caption(f"Algoritmo: {algo} — {choice['reason']}")
    timesteps = st.number_input("Timesteps", value=20000, step=1000)
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
                str(int(timesteps)),
            ]
            try:
                train_drl.main()
            except Exception as e:  # pragma: no cover - user feedback
                st.error(f"Fallo al entrenar: {e}")

        log_box = st.empty()
        thread = threading.Thread(target=_run_train, daemon=True)
        thread.start()
        lines: list[str] = []
        log_iter = log_subscribe(level="info")
        while thread.is_alive():
            try:
                entry = next(log_iter)
                lines.append(f"[{entry['kind']}] {entry['message']}")
                log_box.code("\n".join(lines[-200:]))
            except Exception:
                pass
        thread.join()
        # Drain any remaining log lines
        for _ in range(50):
            try:
                entry = next(log_iter)
                lines.append(f"[{entry['kind']}] {entry['message']}")
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
st.subheader("Actividad en vivo")
kind_options = [
    "trades",
    "riesgo",
    "datos",
    "checkpoints",
    "llm",
    "metricas",
    "reward_tuner",
    "algo_controller",
    "stage_scheduler",
    "dqn_stability",
    "ppo_control",
]
selected_kinds = st.multiselect("Tipos", kind_options, default=kind_options, key="log_kind_sel")

if "log_paused" not in st.session_state:
    st.session_state["log_paused"] = False

if st.button("Pausar" if not st.session_state["log_paused"] else "Reanudar", key="pause_feed"):
    st.session_state["log_paused"] = not st.session_state["log_paused"]

placeholder = st.empty()
if "log_lines" not in st.session_state:
    st.session_state["log_lines"] = []

if "log_iter" not in st.session_state or st.session_state.get("log_iter_kinds") != set(selected_kinds):
    st.session_state["log_iter_kinds"] = set(selected_kinds)
    st.session_state["log_iter"] = log_subscribe(kinds=set(selected_kinds))

if not st.session_state.get("busy") and not st.session_state["log_paused"]:
    start = time.time()
    gen = st.session_state["log_iter"]
    while time.time() - start < 0.5:
        try:
            item = next(gen)
            st.session_state["log_lines"].append(item["message"])
        except StopIteration:
            break
        except Exception:
            break
    st.session_state["log_lines"] = st.session_state["log_lines"][-200:]
placeholder.text("\n".join(st.session_state.get("log_lines", [])))

if not st.session_state.get("busy") and not st.session_state["log_paused"]:
    time.sleep(0.5)
    st.rerun()
