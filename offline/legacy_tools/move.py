"""Utilitar istoric de mutare cache; opereaza pe cai relative la CWD."""

import os
import shutil
import json

def backup_and_replace(converted_file: str, cache_file: str):
    """
    1. Move the existing cache_file into old/ as a backup (if it exists)
    2. Muta/redenumeste converted_file in cache_file
    """
    if not os.path.exists(converted_file):
        raise FileNotFoundError(f"Converted file does not exist: {converted_file}")
    old_folder = "old"
    os.makedirs(old_folder, exist_ok=True)

    # If the cache file exists, move it into old/.
    if os.path.exists(cache_file):
        backup_path = os.path.join(old_folder, os.path.basename(cache_file))
        shutil.move(cache_file, backup_path)
        print(f"[INFO] The old file {cache_file} was moved into {backup_path}")

    # Move the converted file into the cache file's place.
    shutil.move(converted_file, cache_file)
    print(f"[INFO] The converted file {converted_file} was moved/renamed into {cache_file}")


if __name__ == "__main__":
    backup_and_replace("btc_converted.json", "cache_price_BTCUSDC.json")
    backup_and_replace("tao_converted.json", "cache_price_TAOUSDC.json")
