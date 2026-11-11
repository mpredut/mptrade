"""A historical manual converter; it is not part of the runtime."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import normalize_epoch_sec, safe_pos_float
from state_io import atomic_write_json, read_json_file


def _sanitize_price_rows(rows: object) -> list[list[float]]:
    if not isinstance(rows, list):
        return []
    cleaned: list[list[float]] = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts = normalize_epoch_sec(item[0], default=None, allow_zero=True)
        px = safe_pos_float(item[1], default=None)
        if ts is not None and px is not None:
            cleaned.append([ts, px])
    return cleaned


def convert_price_cache(input_file: str, output_file: str, symbol: str):
    data = read_json_file(input_file)

    if not isinstance(data, dict):
        data = {"items": data if isinstance(data, list) else []}

    # Support both legacy {"items": [[ts, price], ...]} and already-converted {"items": {SYMBOL: [...]}}
    raw_items = data.get("items", [])
    if isinstance(raw_items, dict):
        preserved = {
            str(k): _sanitize_price_rows(v)
            for k, v in raw_items.items()
            if isinstance(k, str)
        }
        preserved.setdefault(symbol, [])
        items_map = preserved
    else:
        items_map = {symbol: _sanitize_price_rows(raw_items)}

    new_data = {
        "items": items_map,
        "fetchtime": data.get("fetchtime", {}) if isinstance(data.get("fetchtime"), dict) else {}
    }

    atomic_write_json(output_file, new_data, indent=2, separators=(",", ": "))

    print(f"[INFO] Converted for {symbol} and saved into {output_file}")



if __name__ == "__main__":
    convert_price_cache("cache_price_BTCUSDC.json", "btc_converted.json", "BTCUSDC")
    convert_price_cache("cache_price_TAOUSDC.json", "tao_converted.json", "TAOUSDC")
