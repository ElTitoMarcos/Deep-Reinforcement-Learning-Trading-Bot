import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from src.utils import paths


def test_evaluate_creates_report(tmp_path):
    raw = tmp_path / 'raw'
    rep = tmp_path / 'reports'
    ckpt = tmp_path / 'checkpoints'
    paths.RAW_DIR = raw
    paths.REPORTS_DIR = rep
    paths.CHECKPOINTS_DIR = ckpt
    paths.ensure_dirs_exist()

    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['fees'] = {'taker': 0.005, 'maker': 0.004}
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    data_dir = paths.RAW_DIR / 'binance' / paths.symbol_to_dir('BTC/USDT')
    data_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            'open': [100, 103, 101, 101],
            'high': [100, 103, 101, 101],
            'low': [100, 103, 101, 101],
            'close': [100, 103, 101, 101],
        }
    )
    df.to_csv(data_dir / '1m.csv', index=False)

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    env['DRLTB_RAW_DIR'] = str(raw)
    env['DRLTB_REPORTS_DIR'] = str(rep)
    env['DRLTB_CHECKPOINTS_DIR'] = str(ckpt)
    res = subprocess.run(
        ['python', '-m', 'src.backtest.evaluate', '--config', str(cfg_path), '--policy', 'deterministic'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert '0.005' in res.stdout

    reports_dir = paths.REPORTS_DIR
    runs = [p for p in reports_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    assert (reports_dir / 'experiments.jsonl').is_file()
    assert (run_dir / 'metrics.json').is_file()
    assert (run_dir / 'trades.csv').is_file()
    assert (run_dir / 'equity.csv').is_file()
