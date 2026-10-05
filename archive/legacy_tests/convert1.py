import json


old_file = "cache_price_BTCUSDC.json"
new_file = "cache_price_converted.json"


def convert_price_cache(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        data = json.load(f)

    items = data.get("items", {})
    new_items = {}

    for symbol, records in items.items():
        new_items[symbol] = []
        for rec in records:
            if isinstance(rec, dict):
                new_items[symbol].append([rec["timestamp"], rec["price"]])
            else:
                # Already a list: leave it unchanged.
                new_items[symbol].append(rec)

    new_data = {"items": new_items}

    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Conversion complete: {input_file} -> {output_file}")

convert_price_cache(old_file, new_file)
