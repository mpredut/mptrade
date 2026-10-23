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

from botcore import load_dotenv, required_env, required_float_env, required_int_env
from orchestratorOS.admin.prune_llm_threads import (
    find_latest_project_cid,
    prepare_topic_session,
    prune_old_cli_threads,
    record_topic_turn,
    resolve_agy_bot_bin,
    resolve_bot_cli_base,
    resolve_project_uuid,
    sync_thread_title,
)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.sentiment.gemini_client")

DEFAULT_MODEL = required_env("LLM_DEFAULT_MODEL")
AGY_CLI_PATH = resolve_agy_bot_bin()
AGY_STORE_SYNC_ENABLED = os.environ.get("MPTRADE_AGY_STORE_SYNC", "0") == "1"


DEFAULT_AUTONOMOUS_PROJECT_ID = resolve_project_uuid(required_env("AGY_PROJECT"))


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
        self.cli_path = resolve_agy_bot_bin(cli_path) if cli_path else resolve_agy_bot_bin(AGY_CLI_PATH)
        raw_project = project_id or required_env("AGY_PROJECT")
        self.project_id = resolve_project_uuid(raw_project, fallback_uuid=DEFAULT_AUTONOMOUS_PROJECT_ID)
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
        topic_key: Optional[str] = None,
        conversation_id: Optional[str] = None,
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

        if any(
            os.environ.get(k, "").strip().lower() in ("1", "true", "yes", "on")
            for k in ("DISABLE_GEMINI_LLM", "BACKTEST_MODE", "REPLAY_MODE")
        ):
            logger.debug("Gemini LLM query suppressed in backtest/replay mode.")
            return None

        if (
            "PYTEST_CURRENT_TEST" in os.environ
            and not hasattr(subprocess.run, "_mock_name")
            and not hasattr(urllib.request.urlopen, "_mock_name")
        ):
            logger.debug("Gemini LLM external query suppressed under pytest.")
            return None

        # 2. Local authenticated CLI runner (subscription mode)
        if self.cli_path and os.path.exists(self.cli_path):
            try:
                bot_cli_base = resolve_bot_cli_base()
                resume_cid: Optional[str] = conversation_id
                history_prefix = ""
                if topic_key and not resume_cid and not hasattr(subprocess.run, "_mock_name"):
                    try:
                        resume_cid, history_prefix = prepare_topic_session(
                            topic_key,
                            cli_base=bot_cli_base,
                        )
                    except Exception as e:
                        logger.debug("Failed preparing topic session for %s: %s", topic_key, e)

                effective_prompt = f"{history_prefix}{prompt}" if history_prefix else prompt
                cli_prompt = f"{thread_title}\n\n{effective_prompt}" if thread_title else effective_prompt

                cmd = [
                    self.cli_path,
                    "--project", self.project_id,
                    "--model", selected_model,
                ]
                if resume_cid:
                    cmd.extend(["--conversation", resume_cid])
                cmd.extend(["-p", cli_prompt])

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                # If resuming an existing topic thread failed, fall back cleanly to a fresh thread once
                if (proc.returncode != 0 or not proc.stdout.strip()) and resume_cid and not conversation_id:
                    logger.info("Resuming topic '%s' conversation %s failed; rotating to new thread.", topic_key, resume_cid)
                    resume_cid = None
                    fresh_cmd = [
                        self.cli_path,
                        "--project", self.project_id,
                        "--model", selected_model,
                        "-p", cli_prompt,
                    ]
                    proc = subprocess.run(
                        fresh_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                        check=False,
                    )

                if proc.returncode == 0 and proc.stdout.strip():
                    output = proc.stdout.strip()
                    if cache_ttl_sec > 0:
                        self._cache[prompt] = (now, output)
                    if not hasattr(subprocess.run, "_mock_name"):
                        try:
                            target_cid = resume_cid or find_latest_project_cid(
                                self.project_id,
                                cli_base=bot_cli_base,
                            )
                            if topic_key:
                                record_topic_turn(
                                    topic_key=topic_key,
                                    conversation_id=target_cid,
                                    response_text=output,
                                    thread_title=thread_title,
                                    cli_base=bot_cli_base,
                                )
                            if AGY_STORE_SYNC_ENABLED:
                                if thread_title and target_cid:
                                    sync_thread_title(target_cid, thread_title, cli_base=bot_cli_base)
                                if self.project_id == DEFAULT_AUTONOMOUS_PROJECT_ID:
                                    prune_old_cli_threads(
                                        cli_base=bot_cli_base,
                                        retention_days=required_float_env("CLI_THREAD_RETENTION_DAYS"),
                                        project_id=self.project_id,
                                        all_threads_in_project=False,
                                        max_threads=required_int_env("LLM_MAX_AUTONOMOUS_THREADS"),
                                    )
                        except Exception as e:
                            logger.debug("Failed syncing/pruning CLI thread: %s", e)
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
        topic_key: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Gemini and parse the response strictly as a JSON object."""
        raw_text = self.query_text(
            prompt,
            model=model,
            timeout_sec=timeout_sec,
            cache_ttl_sec=cache_ttl_sec,
            thread_title=thread_title,
            topic_key=topic_key,
            conversation_id=conversation_id,
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
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to decode JSON from Gemini output (raw text: %s)", raw_text[:200])
            return None


# Generic LLM alias
LLMClient = GeminiClient
