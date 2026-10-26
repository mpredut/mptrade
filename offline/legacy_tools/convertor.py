"""A historical manual converter; it is not part of the runtime."""

import json

def convert_price_cache(input_file: str, output_file: str, symbol: str):
    with open(input_file, "r") as f:
        data = json.load(f)

    # input: {"items": [[ts, price], [ts, price], ...]}
    raw_items = data.get("items", [])

    # output: {"items": {SYMBOL: [[ts, price], ...]}, "fetchtime": {}}
    new_data = {
        "items": {
            symbol: raw_items
        },
        "fetchtime": data.get("fetchtime", {})
    }

    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=2, separators=(",", ": "))

    print(f"[INFO] Converted for {symbol} and saved into {output_file}")



if __name__ == "__main__":
    convert_price_cache("cache_price_BTCUSDC.json", "btc_converted.json", "BTCUSDC")
    convert_price_cache("cache_price_TAOUSDC.json", "tao_converted.json", "TAOUSDC")
