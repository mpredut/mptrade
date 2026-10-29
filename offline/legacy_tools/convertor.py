"""A historical manual converter; it is not part of the runtime."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from state_io import atomic_write_json


def convert_price_cache(input_file: str, output_file: str, symbol: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        data = {"items": data if isinstance(data, list) else []}

    # input: {"items": [[ts, price], [ts, price], ...]}
    raw_items = data.get("items", [])

    # output: {"items": {SYMBOL: [[ts, price], ...]}, "fetchtime": {}}
    new_data = {
        "items": {
            symbol: raw_items if isinstance(raw_items, list) else []
        },
        "fetchtime": data.get("fetchtime", {}) if isinstance(data.get("fetchtime"), dict) else {}
    }

    atomic_write_json(output_file, new_data, indent=2, separators=(",", ": "))

    print(f"[INFO] Converted for {symbol} and saved into {output_file}")



if __name__ == "__main__":
    convert_price_cache("cache_price_BTCUSDC.json", "btc_converted.json", "BTCUSDC")
    convert_price_cache("cache_price_TAOUSDC.json", "tao_converted.json", "TAOUSDC")
