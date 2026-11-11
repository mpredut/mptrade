"""Historical cache move utility; operates on paths relative to CWD."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from state_io import durable_replace_file


def backup_and_replace(converted_file: str, cache_file: str):
    """
    1. Move the existing cache_file into old/ as a backup (if it exists)
    2. Move/rename converted_file into cache_file
    """
    if not os.path.exists(converted_file):
        raise FileNotFoundError(f"Converted file does not exist: {converted_file}")
    old_folder = "old"
    os.makedirs(old_folder, exist_ok=True)

    # If the cache file exists, move it into old/.
    if os.path.exists(cache_file):
        backup_path = os.path.join(old_folder, os.path.basename(cache_file))
        durable_replace_file(cache_file, backup_path)
        print(f"[INFO] The old file {cache_file} was moved into {backup_path}")

    # Move the converted file into the cache file's place.
    durable_replace_file(converted_file, cache_file)
    print(f"[INFO] The converted file {converted_file} was moved/renamed into {cache_file}")


if __name__ == "__main__":
    backup_and_replace("btc_converted.json", "cache_price_BTCUSDC.json")
    backup_and_replace("tao_converted.json", "cache_price_TAOUSDC.json")
