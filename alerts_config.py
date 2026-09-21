"""Compatibility wrapper for alerts_config."""
from market_monitor.alerts_config import load_config, resolve, _pair, _SETTING_KEYS, _LIST_KEYS, _BUCKET_ALIAS

__all__ = ["load_config", "resolve", "_pair", "_SETTING_KEYS", "_LIST_KEYS", "_BUCKET_ALIAS"]
