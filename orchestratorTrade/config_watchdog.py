import os
import time
import hashlib
import logging
from typing import Dict, Any

class HotReloadWatchdog:
    def __init__(self, bot_manager):
        self.bot_manager = bot_manager
        self.env_hashes = {}
        self.last_cache_times = {}

    def _hash_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return "
