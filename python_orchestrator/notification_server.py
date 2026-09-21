#!/usr/bin/env python3
import asyncio
import os
import sys
import json
import logging
import re
import time
import requests
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Orchestrator: %(message)s")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class NotificationServer:
    def __init__(self):
        self.ntfy_token = self._load_ntfy_token()
        self.rules = self._load_rules()
        self.cooldowns: Dict[str, float] = {}

    def _load_ntfy_token(self) -> str:
        tok = os.environ.get("NTFY_TOKEN", "").strip()
        if not tok:
            try:
                with open(os.path.expanduser("~/.binance_ntfy_token"), encoding="utf-8") as fh:
                    tok = fh.read().strip()
            except OSError:
                tok = ""
        return tok

    def _load_rules(self) -> List[Dict[str, Any]]:
        rules_path = os.path.join(os.path.dirname(__file__), "notification_rules.json")
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for rule in data.get("rules", []):
                    rule["regex_obj"] = re.compile(rule["match_regex"])
                return data.get("rules", [])
        except Exception as e:
            logging.error(f"Failed to load notification_rules.json: {e}")
            return []

    def _resolve_topic(self, category: str) -> str:
        cat = category.upper()
        # Fallbacks to old naming if env vars are present
        topic = os.environ.get(f"NTFY_TOPIC_{cat}")
        return topic or "test-mptrade"

    def _send_ntfy(self, title: str, message: str, priority: str, topic: str):
        if not topic:
            return
        
        url = f"https://ntfy.sh/{topic}"
        headers = {
            "Title": title.encode("utf-8").decode("latin-1", "ignore"),
            "Priority": priority
        }
        if self.ntfy_token:
            headers["Authorization"] = f"Bearer {self.ntfy_token}"
            
        try:
            # We don't block the main event loop because we want to be fast,
            # but for simplicity in this synchronous method, we use requests with a short timeout.
            # In a fully async refactor we'd use aiohttp or asyncio.to_thread
            requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
        except Exception as e:
            logging.error(f"Failed to send ntfy alert: {e}")

    def process_line(self, line: str, bot_name: str):
        # 1. Check for Explicit AlertNotifier Intent (JSON)
        if "__orchestrator_intent__" in line:
            try:
                payload = json.loads(line)
                if payload.get("__orchestrator_intent__") == "ntfy_webhook":
                    alerts = payload.get("alerts", [])
                    if alerts:
                        first = alerts[0]
                        title = first.get("name", "Alert")
                        body = first.get("body", "")
                        self._send_ntfy(f"[{bot_name}] {title}", body, "high", self._resolve_topic("TRADES"))
                return
            except json.JSONDecodeError:
                pass

        # 2. Check for the [NTFY] prefix shortcut
        if "[NTFY]" in line:
            parts = line.split("[NTFY]", 1)
            if len(parts) > 1:
                msg = parts[1].strip()
                self._send_ntfy(f"[{bot_name}] Alert", msg, "high", self._resolve_topic("ERROR"))
                return

        # 3. Check Intelligent Rules (Regex)
        for rule in self.rules:
            if rule["regex_obj"].search(line):
                # Enforce Cooldown
                rule_key = f"{bot_name}_{rule['name']}"
                now = time.time()
                last_time = self.cooldowns.get(rule_key, 0)
                cooldown_sec = rule.get("cooldown_minutes", 0) * 60
                
                if now - last_time >= cooldown_sec:
                    self.cooldowns[rule_key] = now
                    topic = self._resolve_topic(rule.get("topic_category", "ERROR"))
                    self._send_ntfy(f"[{bot_name}] {rule['name']}", line.strip(), rule.get("priority", "default"), topic)
                break
