import math
import os
import time
import threading

from botcore import _parse_bool_flag, parse_dotenv

# Configuration cache.
config_cache = {}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_ENV_PATH = os.path.join(ROOT_DIR, "config.env")
CONFIG_TXT_PATH = os.path.join(ROOT_DIR, "config.txt")


def _parse_bool(value, default: bool = False) -> bool:
    return _parse_bool_flag(value, default)


def _read_properties_file(path: str) -> dict:
    if not path or isinstance(path, (bool, int)):
        return {}
    try:
        if not os.path.exists(path):
            return {}
        raw = parse_dotenv(path)
        out = {}
        for k, v in raw.items():
            if isinstance(v, str):
                low = v.strip().lower()
                if low in ("true", "1", "yes", "on", "y", "t"):
                    out[k] = True
                elif low in ("false", "0", "no", "off", "n", "f"):
                    out[k] = False
                else:
                    out[k] = v
            else:
                out[k] = v
        return out
    except Exception:
        return {}


def load_config():
    """
    Load configuration from config.env (and legacy config.txt if present) and refresh cache.
    """
    global config_cache
    new_config = {}

    # 1. Load from centralized config.env
    env_file_settings = _read_properties_file(CONFIG_ENV_PATH)
    if "TRADE_ENABLED" in env_file_settings and env_file_settings["TRADE_ENABLED"] != "":
        new_config["trade_enabled"] = _parse_bool(env_file_settings["TRADE_ENABLED"], default=True)
    elif "trade_enabled" in env_file_settings and env_file_settings["trade_enabled"] != "":
        new_config["trade_enabled"] = _parse_bool(env_file_settings["trade_enabled"], default=True)

    # 2. Legacy config.txt if present
    txt_settings = _read_properties_file(CONFIG_TXT_PATH)
    for k, v in txt_settings.items():
        if k.lower() == "trade_enabled":
            if v != "":
                new_config["trade_enabled"] = _parse_bool(v, default=True)
        else:
            new_config[k.lower()] = v

    # 3. Environment variables take highest priority
    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None and env_trade.strip() != "":
        new_config["trade_enabled"] = _parse_bool(env_trade, default=True)

    # Default to true if not specified
    if "trade_enabled" not in new_config:
        new_config["trade_enabled"] = True

    config_cache = new_config


_watcher_stop_event = threading.Event()


def config_watcher(interval=5 * 60):  # 5 minutes
    """
    Periodically monitor configuration and reload the cache.
    """
    wait_sec = 5.0 * 60.0
    if not isinstance(interval, bool) and interval is not None:
        try:
            cand = float(interval)
            if math.isfinite(cand) and cand > 0:
                wait_sec = cand
        except (TypeError, ValueError, OverflowError):
            pass
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
        
