import yaml
from src.utils.config import load_config


def test_defaults_merge(tmp_path):
    # Load current defaults and remove new keys to simulate older config
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        base = yaml.safe_load(f)
    old = base.copy()
    for key in ['reward_tuner', 'hybrid', 'algo_controller', 'stage_scheduler']:
        old.pop(key, None)
    cfg_path = tmp_path / 'old.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(old, f)

    cfg = load_config(str(cfg_path))
    defaults_used = cfg.get('_defaults_used', [])

    assert cfg['reward_tuner']['enabled'] is True
    assert cfg['hybrid']['enabled'] is True and cfg['hybrid']['train_both'] is True
    assert cfg['algo_controller']['llm_enabled'] is False
    assert cfg['stage_scheduler']['enabled'] is True

    for key in ['reward_tuner', 'hybrid', 'algo_controller', 'stage_scheduler']:
        assert key in defaults_used
