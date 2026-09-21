import os
import time
import threading

# Configuration cache.
config_cache = {}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_ENV_PATH = os.path.join(ROOT_DIR, "config.env")
CONFIG_TXT_PATH = os.path.join(ROOT_DIR, "config.txt")


def _read_properties_file(path: str) -> dict:
    result = {}
    if not os.path.exists(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().split("#", 1)[0].strip()
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
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
        new_config["trade_enabled"] = bool(env_file_settings["TRADE_ENABLED"])
    elif "trade_enabled" in env_file_settings:
        new_config["trade_enabled"] = bool(env_file_settings["trade_enabled"])

    # 2. Legacy config.txt if present
    txt_settings = _read_properties_file(CONFIG_TXT_PATH)
    for k, v in txt_settings.items():
        new_config[k.lower()] = v

    # 3. Environment variables take highest priority
    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None:
        new_config["trade_enabled"] = env_trade.strip().lower() in ("true", "1", "yes")

    # Default to true if not specified
    if "trade_enabled" not in new_config:
        new_config["trade_enabled"] = True

    config_cache = new_config


def config_watcher(interval=5 * 60):  # 5 minutes
    """
    Periodically monitor configuration and reload the cache.
    """
    while True:
        load_config()
        time.sleep(interval)


def is_trade_enabled():
    """
    Return whether cached ``trade_enabled`` is true.
    """
    return bool(config_cache.get("trade_enabled", True))


watcher_thread = None

def start_config_watcher():
    global watcher_thread

    if watcher_thread and watcher_thread.is_alive():
        return

    watcher_thread = threading.Thread(
        target=config_watcher,
        name="start_config_watcher",
        daemon=True
    )
    watcher_thread.start()
    print("Config watcher started.")


def stop_config_watcher():
    global watcher_thread
    if watcher_thread:
        watcher_thread.join()
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
        
