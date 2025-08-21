from utils.credentials import load_env
from exchange.binance_factory import (
    make_public_mainnet, make_private_mainnet, make_private_testnet
)


def resolve_clients():
    cfg = load_env()
    rl = max(1, int(60000 / max(1, cfg["rate_limit_per_min"])))
    public = make_public_mainnet(rl)

    private_main = None
    private_test = None
    if cfg["mainnet_key"] and cfg["mainnet_sec"]:
        private_main = make_private_mainnet(cfg["mainnet_key"], cfg["mainnet_sec"], rl)
    if cfg["testnet_key"] and cfg["testnet_sec"]:
        private_test = make_private_testnet(cfg["testnet_key"], cfg["testnet_sec"], rl)

    return {"public_mainnet": public, "private_mainnet": private_main, "private_testnet": private_test}
