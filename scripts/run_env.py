from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
import os
_DOTENV = find_dotenv(usecwd=True)
load_dotenv(_DOTENV, override=True)
if __name__ == "__main__" or os.getenv("DEBUG_DOTENV") == "1":
    print(f"[.env] Cargado: {_DOTENV or 'NO ENCONTRADO'}")

import argparse

from src.utils.config import load_config
from src.policies.router import get_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trading env with selected policy")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--data-mode", dest="data_mode", help="Feature set descriptor")
    parser.add_argument("--policy", dest="policy_override", help="Manually override policy type")
    args = parser.parse_args()

    overrides = {"data_mode": args.data_mode, "policy_override": args.policy_override}
    cfg = load_config(args.config, overrides=overrides)
    data_mode = cfg.get("data_mode", "price_only")
    policy = get_policy(data_mode, policy_override=cfg.get("policy_override"))
    print(f"Selected policy {policy.__class__.__name__} for data_mode='{data_mode}'")


if __name__ == "__main__":
    main()
