"""Google Gemini client providing LLM reasoning via authenticated CLI or REST API."""
from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import urllib.request

from botcore import load_dotenv, required_env
from orchestratorOS.agy.paths import resolve_agy_bot_bin, resolve_bot_cli_base
from orchestratorOS.agy.topics import (
    TopicSession,
    extract_json_object,
    format_topic_history_prefix,
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
        timeout_f = 15.0
        if timeout_sec is not None and not isinstance(timeout_sec, bool):
            try:
                tf = float(timeout_sec)
                if math.isfinite(tf) and tf > 0.0:
                    timeout_f = tf
            except (TypeError, ValueError, OverflowError):
                pass
        timeout_sec = timeout_f
        ttl_f = 0.0
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                cf = float(cache_ttl_sec)
                if math.isfinite(cf) and cf > 0.0:
                    ttl_f = cf
            except (TypeError, ValueError, OverflowError):
                pass
        cache_ttl_sec = ttl_f

        if cache_ttl_sec > 0:
            cached = self._cache.get(prompt)
            if cached and 0.0 <= (now - cached[0]) < cache_ttl_sec:
                return cached[1]

        # 1. Custom runner (used for unit tests)
        if self._custom_runner is not None:
            try:
                res = self._custom_runner(prompt, selected_model, timeout_sec)
                if cache_ttl_sec > 0 and res:
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
                is_mocked = hasattr(subprocess.run, "_mock_name")
                use_topic_session = bool(
                    topic_key and (not is_mocked or hasattr(prepare_topic_session, "_mock_name"))
                )
                session = TopicSession(conversation_id, "", False)
                if use_topic_session and not conversation_id:
                    session = prepare_topic_session(topic_key, cli_base=bot_cli_base)

                # The per-call title is a turn header; the thread itself keeps its permanent topic name.
                effective_title = format_topic_turn_header(topic_key, thread_title) if topic_key else thread_title
                body = f"{session.history_prefix}{prompt}" if session.history_prefix else prompt
                cli_prompt = f"{effective_title}\n\n{body}" if effective_title else body

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
                    if is_mocked or not os.path.isdir(bot_cli_base):
                        return subprocess.run(cmd, capture_output=True, text=True, timeout=call_timeout,
                                              check=False, env=run_env)
                    lock_path = os.path.join(bot_cli_base, ".agy_bot_cli.lock")
                    lock_wait = max(90.0, call_timeout) if call_timeout >= 30.0 else min(call_timeout * 0.5, 5.0)
                    deadline = time.monotonic() + lock_wait
                    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
                        locked = False
                        while True:
                            try:
                                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                locked = True
                                break
                            except BlockingIOError:
                                if time.monotonic() >= deadline:
                                    break
                                time.sleep(0.15)
                        try:
                            return subprocess.run(cmd, capture_output=True, text=True, timeout=call_timeout,
                                                  check=False, env=run_env)
                        finally:
                            if locked:
                                try:
                                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                                except OSError:
                                    pass

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
                    raw_resp = payload.get("response")
                    text_out = raw_resp.strip() if isinstance(raw_resp, str) else ""
                    raw_cid = payload.get("conversation_id")
                    cid_out = raw_cid.strip() if isinstance(raw_cid, str) and raw_cid.strip() else None
                    return (text_out or None), cid_out

                output, turn_cid = None, None
                proc = None
                try:
                    proc = _run_cli(session.conversation_id, cli_prompt, timeout_sec)
                    output, turn_cid = _parse(proc)
                except subprocess.TimeoutExpired:
                    if not (session.conversation_id and not conversation_id):
                        raise
                    logger.warning(
                        "Resuming topic '%s' conversation %s timed out after %ss; falling back to one-off thread.",
                        topic_key,
                        session.conversation_id,
                        timeout_sec,
                    )

                if output is None and session.conversation_id and not conversation_id:
                    # Runtime resilience: answer this call from a detached one-off thread. The topic keeps its
                    # permanent conversation; the next call resumes it again.
                    logger.info("Resuming topic '%s' conversation %s failed; answering from a one-off thread.",
                                topic_key, session.conversation_id)
                    fallback_prompt = cli_prompt
                    if use_topic_session and topic_key and not session.history_prefix:
                        try:
                            hist_prefix = format_topic_history_prefix(topic_key)
                            if hist_prefix:
                                fallback_body = f"{hist_prefix}{prompt}"
                                fallback_prompt = f"{effective_title}\n\n{fallback_body}" if effective_title else fallback_body
                        except Exception:
                            pass
                    proc = _run_cli(None, fallback_prompt, timeout_sec)
                    output, detached_cid = _parse(proc)
                    if detached_cid and detached_cid != session.conversation_id and not is_mocked:
                        self._remove_detached_thread(bot_cli_base, detached_cid)
                    turn_cid = None

                if output:
                    if cache_ttl_sec > 0:
                        self._cache[prompt] = (now, output)
                    if use_topic_session:
                        def _finalize_topic_turn() -> None:
                            compacted_ok = False
                            if session.compact and turn_cid:
                                compact_timeout = max(timeout_sec, 120.0)
                                try:
                                    c_proc = _run_cli(
                                        turn_cid,
                                        "/compact Summarize prior turns in place, preserving open positions, recent decisions, and active risk guards.",
                                        compact_timeout,
                                    )
                                    c_out, c_cid = _parse(c_proc)
                                    compacted_ok = bool(c_out or c_cid)
                                except Exception as exc:
                                    logger.warning(
                                        "In-place /compact for topic '%s' conversation %s failed: %s",
                                        topic_key,
                                        turn_cid,
                                        exc,
                                    )
                            try:
                                record_topic_turn(
                                    topic_key=topic_key,
                                    conversation_id=turn_cid,
                                    response_text=output,
                                    compacted=compacted_ok,
                                    cli_base=bot_cli_base,
                                )
                            except Exception as e:
                                logger.debug("Failed recording topic turn for %s: %s", topic_key, e)

                        if session.compact and turn_cid and not is_mocked:
                            threading.Thread(target=_finalize_topic_turn, daemon=False).start()
                        else:
                            _finalize_topic_turn()
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

    @staticmethod
    def _remove_detached_thread(bot_cli_base: str, cid: str) -> None:
        """Delete a one-off fallback conversation from the isolated bot store so orphans never accumulate."""
        import sqlite3

        if not bot_cli_base or not cid:
            return
        conv_db = os.path.join(bot_cli_base, "conversations", f"{cid}.db")
        for sfx in ("", "-wal", "-shm"):
            try:
                if os.path.exists(conv_db + sfx):
                    os.remove(conv_db + sfx)
            except OSError:
                pass
        ann_path = os.path.join(bot_cli_base, "annotations", f"{cid}.pbtxt")
        try:
            if os.path.exists(ann_path):
                os.remove(ann_path)
        except OSError:
            pass
        shutil.rmtree(os.path.join(bot_cli_base, "brain", cid), ignore_errors=True)
        sum_db = os.path.join(bot_cli_base, "conversation_summaries.db")
        if os.path.exists(sum_db):
            try:
                conn = sqlite3.connect(sum_db, timeout=5)
                try:
                    conn.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (cid,))
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error:
                pass

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
        if isinstance(raw_text, dict):
            return raw_text

        parsed = extract_json_object(str(raw_text))
        if parsed is None:
            logger.warning("Failed to decode JSON from Gemini output (raw text: %s)", str(raw_text)[:200])
        return parsed


# Generic LLM alias
LLMClient = GeminiClient
