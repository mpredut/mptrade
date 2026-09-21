import os
import time
import threading

# Configuration cache.
config_cache = {}

# Configuration file path.
config_file_path = "config.txt"

def load_config():
    """
    Load the configuration file and refresh the cache.
    """
    global config_cache
    new_config = {}

    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None:
        new_config["trade_enabled"] = env_trade.strip().lower() in ("true", "1", "yes")

    try:
        with open(config_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Convert textual true/false values to booleans.
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    new_config[key] = value
            config_cache = new_config
    except FileNotFoundError:
        if "trade_enabled" not in new_config:
            new_config["trade_enabled"] = True
        config_cache = new_config

def config_watcher(interval= 5 * 60): # 5 minutes
    """
    Periodically monitor the configuration file and reload the cache.
    """
    while True:
        load_config()
        time.sleep(interval)

def is_trade_enabled():
    """
    Return whether cached ``trade_enabled`` is true.
    """
    return config_cache.get("trade_enabled", False)


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
        
