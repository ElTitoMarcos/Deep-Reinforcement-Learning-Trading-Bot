#!/usr/bin/env bash
set -euo pipefail
VENV_ACTIVATE="D:\Descargas\drl_trading_base_with_ui_fixed/.venv/bin/activate"
if ! grep -q "" ~/.bashrc; then
  echo "source " >> ~/.bashrc
fi
