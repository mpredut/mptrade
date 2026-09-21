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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class NotificationServer:
    def __init__(self):
        self.ntfy_token = self._load_ntfy_token()
        self.rules = self._load_rules()
        self.cooldowns: Dict[str, float] = {}
        
        # Persistent Queue File
        self.queue_file = os.path.join(ROOT_DIR, "cachedb", "notification_queue.jsonl")
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)

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
        rules_path = os.path.join(os.path.dirname(__file__), "rules.json")
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for rule in data.get("rules", []):
                    rule["regex_obj"] = re.compile(rule["match_regex"])
                return data.get("rules", [])
        except Exception as e:
            logging.error(f"Failed to load rules.json: {e}")
            return []

    def _resolve_topic(self, category: str) -> str:
        cat = category.upper()
        # Fallbacks to old naming if env vars are present
        topic = os.environ.get(f"NTFY_TOPIC_{cat}")
        return topic or "test-mptrade"
        
    def _enqueue_payload(self, intent_type: str, data: dict):
        try:
            payload = {"__orchestrator_intent__": intent_type}
            payload.update(data)
            with open(self.queue_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logging.error(f"Failed to enqueue notification: {e}")

    def _send_ntfy(self, title: str, message: str, priority: str, topic: str, is_retry: bool = False) -> bool:
        if not topic:
            return True
        
        url = f"https://ntfy.sh/{topic}"
        headers = {
            "Title": title.encode("utf-8").decode("latin-1", "ignore"),
            "Priority": priority
        }
        if self.ntfy_token:
            headers["Authorization"] = f"Bearer {self.ntfy_token}"
            
        try:
            resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            if not is_retry:
                logging.error(f"Failed to send ntfy alert (queueing): {e}")
                self._enqueue_payload("ntfy_webhook", {
                    "title": title,
                    "message": message,
                    "priority": priority,
                    "topic": topic
                })
            return False
            
    def _send_email(self, subject: str, message: str, is_retry: bool = False) -> bool:
        # Placeholder for actual email delivery. Can be implemented with smtplib.
        # For now, we simulate success if network is up, or queue it.
        # To test network, we just do a quick ping or assume success.
        logging.warning(f"Email delivery not fully configured. Intent recorded for: {subject}")
        return True

    def process_line(self, line: str, bot_name: str):
        # 1. Check for Explicit AlertNotifier Intent (JSON)
        if "__orchestrator_intent__" in line:
            try:
                payload = json.loads(line)
                intent = payload.get("__orchestrator_intent__")
                if intent == "ntfy_webhook":
                    # Direct retry payload
                    if "title" in payload:
                        self._send_ntfy(payload["title"], payload["message"], payload["priority"], payload["topic"])
                        return
                        
                    # Standard alert formatting
                    alerts = payload.get("alerts", [])
                    if alerts:
                        first = alerts[0]
                        title = first.get("name", "Alert")
                        body = first.get("body", "")
                        self._send_ntfy(f"[{bot_name}] {title}", body, "high", self._resolve_topic("TRADES"))
                
                elif intent == "email":
                    subject = payload.get("subject", "Alert")
                    self._send_email(subject, str(payload.get("alerts", [])))
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

    async def flush_queue_loop(self):
        """Background task to resend delayed notifications."""
        while True:
            await asyncio.sleep(60)
            if not os.path.exists(self.queue_file):
                continue
            
            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                remaining = []
                success_count = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        payload = json.loads(line)
                    except:
                        continue
                    
                    if not remaining: # No failures in this batch yet
                        intent = payload.get("__orchestrator_intent__")
                        success = False
                        if intent == "ntfy_webhook":
                            title = payload.get("title", "")
                            msg = payload.get("message", "")
                            if not title.startswith("[DELAYED]"):
                                title = "[DELAYED] " + title
                            success = await asyncio.to_thread(
                                self._send_ntfy,
                                title,
                                msg,
                                payload.get("priority", "default"),
                                payload.get("topic", "test-mptrade"),
                                True
                            )
                        elif intent == "email":
                            success = await asyncio.to_thread(
                                self._send_email,
                                "[DELAYED] " + payload.get("subject", ""),
                                payload.get("message", ""),
                                True
                            )
                        else:
                            success = True # Unknown intent, drop it
                            
                        if success:
                            success_count += 1
                        else:
                            remaining.append(line)
                    else:
                        remaining.append(line)
                        
                if success_count > 0 or len(remaining) != len(lines):
                    with open(self.queue_file, "w", encoding="utf-8") as f:
                        for r in remaining:
                            f.write(r + "\n")
                    if success_count > 0:
                        logging.info(f"Flushed {success_count} delayed notifications from queue.")

            except Exception as e:
                logging.error(f"Error flushing notification queue: {e}")
