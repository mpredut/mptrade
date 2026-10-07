"""Google Gemini client providing LLM reasoning via authenticated CLI or REST API."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional
import urllib.request

logger = logging.getLogger("intelligence.sentiment.gemini_client")

DEFAULT_MODEL = "gemini-3.8-flash-low"
AGY_CLI_PATH = "/home/predut/.local/bin/agy"


DEFAULT_AUTONOMOUS_PROJECT_ID = "f5f9d01f-01e5-4fac-80f2-97364688afab"


class GeminiClient:
    """Wrapper for querying LLM models.

    Prefers the local authenticated `agy` CLI (leveraging active user subscription)
    with fallback to Google AI Studio REST API if an API key is configured.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        cli_path: Optional[str] = None,
        custom_runner: Optional[Callable[[str, str, float], str]] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.cli_path = cli_path or (AGY_CLI_PATH if os.path.exists(AGY_CLI_PATH) else shutil.which("agy"))
        raw_project = project_id or os.environ.get("AGY_PROJECT", "autonommptrade")
        if raw_project in ("autonommptrade", DEFAULT_AUTONOMOUS_PROJECT_ID):
            self.project_id = DEFAULT_AUTONOMOUS_PROJECT_ID
        else:
            self.project_id = raw_project
        self._custom_runner = custom_runner
        self._cache: Dict[str, tuple[float, str]] = {}

    def query_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        timeout_sec: float = 15.0,
        cache_ttl_sec: float = 0.0,
        thread_title: Optional[str] = None,
    ) -> Optional[str]:
        """Query Gemini with a prompt and return the text response."""
        now = time.time()
        selected_model = model or self.model

        if cache_ttl_sec > 0:
            cached = self._cache.get(prompt)
            if cached and (now - cached[0]) < cache_ttl_sec:
                return cached[1]

        # 1. Custom runner (used for unit tests)
        if self._custom_runner is not None:
            try:
                res = self._custom_runner(prompt, selected_model, timeout_sec)
                if cache_ttl_sec > 0:
                    self._cache[prompt] = (now, res)
                return res
            except Exception as e:
                logger.error("Custom runner failed: %s", e)
                return None

        # 2. Local authenticated CLI runner (subscription mode)
        if self.cli_path and os.path.exists(self.cli_path):
            try:
                cli_prompt = f"{thread_title}\n\n{prompt}" if thread_title else prompt
                cmd = [
                    self.cli_path,
                    "--project", self.project_id,
                    "--model", selected_model,
                    "-p", cli_prompt,
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    output = proc.stdout.strip()
                    if cache_ttl_sec > 0:
                        self._cache[prompt] = (now, output)
                    if thread_title:
                        try:
                            from orchestratorOS.admin.prune_autonommptrade import sync_thread_title
                            presence_dir = os.path.expanduser("~/.gemini/antigravity-cli/presence")
                            if os.path.isdir(presence_dir):
                                locks = [
                                    os.path.splitext(f)[0]
                                    for f in os.listdir(presence_dir)
                                    if f.endswith(".lock")
                                ]
                                if locks:
                                    locks.sort(
                                        key=lambda cid: os.path.getmtime(os.path.join(presence_dir, f"{cid}.lock")),
                                        reverse=True,
                                    )
                                    sync_thread_title(locks[0], thread_title)
                        except Exception as e:
                            logger.debug("Failed syncing thread title: %s", e)
                    return output
                else:
                    logger.warning(
                        "agy CLI exited with code %s: %s",
                        proc.returncode,
                        proc.stderr.strip()[:200],
                    )
            except subprocess.TimeoutExpired:
                logger.warning("Gemini query timed out after %s seconds", timeout_sec)
                return None
            except Exception as e:
                logger.warning("Failed to invoke agy CLI: %s", e)

        # 3. Direct REST API fallback if API key is provided
        if self.api_key:
            try:
                clean_model = selected_model.replace("-low", "").replace("-medium", "").replace("-high", "")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={self.api_key}"
                body = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if cache_ttl_sec > 0:
                        self._cache[prompt] = (now, text)
                    return text
            except Exception as e:
                logger.error("Gemini REST API error: %s", e)

        return None

    def query_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        timeout_sec: float = 15.0,
        cache_ttl_sec: float = 0.0,
        thread_title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Gemini and parse the response strictly as a JSON object."""
        raw_text = self.query_text(
            prompt,
            model=model,
            timeout_sec=timeout_sec,
            cache_ttl_sec=cache_ttl_sec,
            thread_title=thread_title,
        )
        if not raw_text:
            return None

        # Clean code fence markdown block if present
        cleaned = raw_text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.warning("Failed to decode JSON from Gemini output: %s (raw text: %s)", err, raw_text[:200])
            return None


# Generic LLM alias
LLMClient = GeminiClient
