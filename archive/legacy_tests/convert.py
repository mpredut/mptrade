import json

old_file = "cache_price_TAOUSDC.json"
new_file = "cache_price_converted.json"
symbol = "BTCUSDT"  # aici setezi simbolul

with open(old_file, "r") as f:
    old_data = json.load(f)

old_items = old_data.get("items", [])

# Turn the list of lists into a list of dicts.
new_items = [
    {"timestamp": ts, "price": price}
    for ts, price in old_items
]

# Determine the last timestamp used for fetchtime.
last_ts = new_items[-1]["timestamp"] if new_items else 0

new_data = {
    "items": {
        symbol: new_items
    },
    "fetchtime": {
        symbol: last_ts
    }
}

with open(new_file, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"Conversion complete -> {new_file}")

