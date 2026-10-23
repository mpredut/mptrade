"""Google Gemini client providing LLM reasoning via authenticated CLI or REST API."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Tuple
import urllib.request

from botcore import load_dotenv, required_env
from orchestratorOS.agy.paths import resolve_agy_bot_bin, resolve_bot_cli_base
from orchestratorOS.agy.topics import (
    TopicSession,
    extract_json_object,
    format_topic_turn_header,
    prepare_topic_session,
    record_topic_turn,
)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.sentiment.gemini_client")

DEFAULT_MODEL = required_env("LLM_DEFAULT_MODEL")
AGY_CLI_PATH = resolve_agy_bot_bin()


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
        self.cli_path = resolve_agy_bot_bin(cli_path) if cli_path else AGY_CLI_PATH
        # agy resolves `--project` by name or id; the configured name is the SSOT.
        self.project_id = project_id or required_env("AGY_PROJECT")
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
                session = TopicSession(conversation_id, "", False)
                if topic_key and not conversation_id and not hasattr(subprocess.run, "_mock_name"):
                    session = prepare_topic_session(topic_key, cli_base=bot_cli_base)

                # The per-call title is a turn header; the thread itself keeps its permanent topic name.
                effective_title = format_topic_turn_header(topic_key, thread_title) if topic_key else thread_title
                body = f"{session.history_prefix}{prompt}" if session.history_prefix else prompt
                cli_prompt = f"{effective_title}\n\n{body}" if effective_title else body
                if session.compact:
                    cli_prompt = f"/compact {cli_prompt}"

                run_env = os.environ.copy()
                run_env["AGY_BOT_HOME"] = os.path.dirname(os.path.dirname(bot_cli_base))

                def _run_cli(
                    resume: Optional[str], text: str, call_timeout: float = timeout_sec
                ) -> subprocess.CompletedProcess:
                    cmd = [self.cli_path, "--project", self.project_id, "--model", selected_model,
                           "--output-format", "json"]
                    if resume:
                        cmd.extend(["--conversation", resume])
                    cmd.extend(["-p", text])
                    return subprocess.run(cmd, capture_output=True, text=True, timeout=call_timeout,
                                          check=False, env=run_env)

                def _parse(proc: subprocess.CompletedProcess) -> Tuple[Optional[str], Optional[str]]:
                    raw = (proc.stdout or "").strip()
                    if proc.returncode != 0 or not raw:
                        return None, None
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        return raw, None
                    if not isinstance(payload, dict):
                        return raw, None
                    text_out = str(payload.get("response") or "").strip()
                    return (text_out or None), (payload.get("conversation_id") or None)

                is_mocked = hasattr(subprocess.run, "_mock_name")
                first_timeout = max(timeout_sec, 90.0) if (session.compact and not is_mocked) else timeout_sec
                output, turn_cid = None, None
                proc = None
                try:
                    proc = _run_cli(session.conversation_id, cli_prompt, first_timeout)
                    output, turn_cid = _parse(proc)
                except subprocess.TimeoutExpired:
                    if not (session.conversation_id and not conversation_id):
                        raise
                    logger.warning(
                        "Resuming topic '%s' conversation %s timed out after %ss; falling back to one-off thread.",
                        topic_key,
                        session.conversation_id,
                        first_timeout,
                    )

                if output is None and session.conversation_id and not conversation_id:
                    # Runtime resilience: answer this call from a detached one-off thread. The topic keeps its
                    # permanent conversation; the next call resumes it again.
                    logger.info("Resuming topic '%s' conversation %s failed; answering from a one-off thread.",
                                topic_key, session.conversation_id)
                    proc = _run_cli(None, cli_prompt.removeprefix("/compact "), timeout_sec)
                    output, _ = _parse(proc)
                    turn_cid = None

                if output:
                    if cache_ttl_sec > 0:
                        self._cache[prompt] = (now, output)
                    if topic_key and not hasattr(subprocess.run, "_mock_name"):
                        try:
                            record_topic_turn(
                                topic_key=topic_key,
                                conversation_id=turn_cid,
                                response_text=output,
                                compacted=session.compact and turn_cid is not None,
                                cli_base=bot_cli_base,
                            )
                        except Exception as e:
                            logger.debug("Failed recording topic turn for %s: %s", topic_key, e)
                    return output
                else:
                    logger.warning(
                        "agy CLI exited with code %s: %s",
                        proc.returncode if proc is not None else "timeout",
                        (proc.stderr or "").strip()[:200] if proc is not None else "",
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

        parsed = extract_json_object(raw_text)
        if parsed is None:
            logger.warning("Failed to decode JSON from Gemini output (raw text: %s)", raw_text[:200])
        return parsed


# Generic LLM alias
LLMClient = GeminiClient
