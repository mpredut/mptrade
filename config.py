import math
import os
import time
import threading

# Configuration cache.
config_cache = {}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_ENV_PATH = os.path.join(ROOT_DIR, "config.env")
CONFIG_TXT_PATH = os.path.join(ROOT_DIR, "config.txt")


def _parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return bool(default)
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1].strip()
        lowered = cleaned.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
        return bool(default)
    return bool(default)


def _read_properties_file(path: str) -> dict:
    result = {}
    if not os.path.exists(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    if line.startswith("export ") or line.startswith("export\t"):
                        line = line[6:].strip()
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().split("#", 1)[0].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                        value = value[1:-1].strip()
                    if value.lower() in ("true", "1", "yes", "on"):
                        value = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        value = False
                    result[key] = value
    except Exception:
        pass
    return result


def load_config():
    """
    Load configuration from config.env (and legacy config.txt if present) and refresh cache.
    """
    global config_cache
    new_config = {}

    # 1. Load from centralized config.env
    env_file_settings = _read_properties_file(CONFIG_ENV_PATH)
    if "TRADE_ENABLED" in env_file_settings:
        new_config["trade_enabled"] = _parse_bool(env_file_settings["TRADE_ENABLED"])
    elif "trade_enabled" in env_file_settings:
        new_config["trade_enabled"] = _parse_bool(env_file_settings["trade_enabled"])

    # 2. Legacy config.txt if present
    txt_settings = _read_properties_file(CONFIG_TXT_PATH)
    for k, v in txt_settings.items():
        if k.lower() == "trade_enabled":
            new_config["trade_enabled"] = _parse_bool(v)
        else:
            new_config[k.lower()] = v

    # 3. Environment variables take highest priority
    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None:
        new_config["trade_enabled"] = _parse_bool(env_trade)

    # Default to true if not specified
    if "trade_enabled" not in new_config:
        new_config["trade_enabled"] = True

    config_cache = new_config


_watcher_stop_event = threading.Event()


def config_watcher(interval=5 * 60):  # 5 minutes
    """
    Periodically monitor configuration and reload the cache.
    """
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0
    ):
        wait_sec = 5.0 * 60.0
    else:
        wait_sec = float(interval)
    while not _watcher_stop_event.is_set():
        load_config()
        if _watcher_stop_event.wait(wait_sec):
            break


def is_trade_enabled():
    """
    Return whether cached ``trade_enabled`` is true.
    """
    return _parse_bool(config_cache.get("trade_enabled", True), default=True)


watcher_thread = None

def start_config_watcher():
    global watcher_thread

    if watcher_thread and watcher_thread.is_alive():
        return

    _watcher_stop_event.clear()
    watcher_thread = threading.Thread(
        target=config_watcher,
        name="start_config_watcher",
        daemon=True
    )
    watcher_thread.start()
    print("Config watcher started.")


def stop_config_watcher():
    global watcher_thread
    _watcher_stop_event.set()
    if watcher_thread:
        watcher_thread.join(timeout=2.0)
        watcher_thread = None
        print("Config watcher stopped.")


load_config()

# A usage example
if __name__ == "__main__":
    start_config_watcher()
    print("Watching the configuration file...")
    try:
        while True:
            # Demonstrate the trading-enabled check.
            print("Trade Enabled:", is_trade_enabled())
            time.sleep(10)
    except KeyboardInterrupt:
        print("Monitoring stopped.")
        stop_config_watcher()
        
