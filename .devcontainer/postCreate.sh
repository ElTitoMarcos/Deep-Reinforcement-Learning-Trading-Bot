#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip wheel
pip install -e .
pip install "torch>=2.2" "stable-baselines3>=2.3.2"
pip install pytest

echo "✅ Entorno listo."
