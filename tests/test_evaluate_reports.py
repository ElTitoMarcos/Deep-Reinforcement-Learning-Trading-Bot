import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def test_evaluate_creates_report(tmp_path):
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['fees'] = {'taker': 0.005, 'maker': 0.004}
    cfg_paths = cfg.setdefault('paths', {})
    cfg_paths['raw_dir'] = str(tmp_path / 'raw')
    cfg_paths['reports_dir'] = str(tmp_path / 'reports')
    cfg_paths['checkpoints_dir'] = str(tmp_path / 'checkpoints')
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    data_dir = Path(cfg_paths['raw_dir']) / 'binance' / 'BTC-USDT'
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
    res = subprocess.run(
        ['python', '-m', 'src.backtest.evaluate', '--config', str(cfg_path), '--policy', 'deterministic'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert '0.005' in res.stdout

    reports_dir = Path(cfg_paths['reports_dir'])
    runs = list(reports_dir.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / 'metrics.json').is_file()
    assert (run_dir / 'trades.csv').is_file()
    assert (run_dir / 'equity.csv').is_file()
