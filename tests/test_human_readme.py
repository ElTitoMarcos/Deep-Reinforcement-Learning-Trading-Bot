from pathlib import Path
from src.reports.human_friendly import write_readme

def test_write_readme_creates_file(tmp_path: Path):
    metrics = {
        "pnl": 0.1,
        "max_drawdown": 0.05,
        "sharpe": 1.2,
        "hit_ratio": 0.6,
        "turnover": 2.0,
    }
    write_readme(metrics, tmp_path)
    readme = tmp_path / "README.md"
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    assert "Ganancia total" in content
    assert "Caída máxima" in content
    assert "Consistencia" in content
