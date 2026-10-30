"""Google Gemini client providing LLM reasoning via authenticated CLI or REST API."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import urllib.request

from botcore import _parse_bool_flag, load_dotenv, required_env
from intelligence.macro_sentiment_ssot import is_cache_fresh, safe_pos_float
from orchestratorOS.agy.paths import resolve_agy_bot_bin, resolve_bot_cli_base
from orchestratorOS.agy.summary_store import delete_threads
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
FALLBACK_MODELS_RAW = required_env("LLM_FALLBACK_MODELS")
DEFAULT_FALLBACK_MODELS = tuple(
    m.strip() for m in FALLBACK_MODELS_RAW.split(",") if m.strip()
)
AGY_CLI_PATH = resolve_agy_bot_bin()


def canonical_rest_model(model_name: str) -> str:
    """Map arbitrary LLM model names (Gemini, Argon, GPT-OSS, Claude) to a canonical Google AI Studio REST model."""
    if not isinstance(model_name, str) or not model_name.strip():
        return "gemini-2.5-flash"
    cleaned = model_name.strip().lower()
    base = cleaned.replace("-low", "").replace("-medium", "").replace("-high", "")
    if base.startswith(("gemini-2.5", "gemini-2.0", "gemini-1.5")):
        return base
    return "gemini-2.5-flash"


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
        fallback_models: Optional[tuple[str, ...] | list[str]] = None,
    ) -> None:
        self.model = model.strip() if isinstance(model, str) and model.strip() else DEFAULT_MODEL
        if fallback_models is not None:
            self.fallback_models = tuple(
                m.strip() for m in fallback_models if isinstance(m, str) and m.strip()
            )
        else:
            self.fallback_models = DEFAULT_FALLBACK_MODELS
        raw_key = (
            api_key
            if isinstance(api_key, str) and api_key.strip()
            else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")
        )
        self.api_key = raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None
        self.cli_path = resolve_agy_bot_bin(cli_path) if cli_path else AGY_CLI_PATH
        # agy resolves `--project` by name or id; the configured name is the SSOT.
        self.project_id = (
            project_id.strip()
            if isinstance(project_id, str) and project_id.strip()
            else required_env("AGY_PROJECT")
        )
        self._custom_runner = custom_runner
        self._cache: Dict[str, tuple[float, str]] = {}

    def get_model_candidates(self, primary_model: Optional[str] = None) -> list[str]:
        """Return an ordered, deduplicated sequence of models to attempt (primary -> fallbacks)."""
        selected = (primary_model.strip() if isinstance(primary_model, str) and primary_model.strip() else None) or self.model
        candidates = [selected]
        for fb in self.fallback_models:
            if fb and fb not in candidates:
                candidates.append(fb)
        return candidates

    def _execute_cli(
        self,
        resume: Optional[str],
        text: str,
        call_timeout: float,
        cli_model: str,
        bot_cli_base: str,
        run_env: dict[str, str],
        is_mocked: bool,
    ) -> subprocess.CompletedProcess:
        """Run the isolated agy CLI tool with robust file-locking and timeout management."""
        cmd = [self.cli_path, "--project", self.project_id, "--model", cli_model, "--output-format", "json"]
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

    @staticmethod
    def _parse_cli_payload(proc: subprocess.CompletedProcess) -> Tuple[Optional[str], Optional[str]]:
        """Extract response text and conversation_id from agy CLI JSON output."""
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
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        now = time.time()
        selected_model = (model.strip() if isinstance(model, str) and model.strip() else None) or self.model
        timeout_sec = safe_pos_float(timeout_sec, 15.0)
        cache_ttl_sec = safe_pos_float(cache_ttl_sec, 0.0)

        if cache_ttl_sec > 0:
            cached = self._cache.get(prompt)
            if isinstance(cached, (tuple, list)) and len(cached) >= 2 and is_cache_fresh(cached[0], now_ts=now, ttl_sec=cache_ttl_sec):
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
            _parse_bool_flag(os.environ.get(k))
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

                # Attempt primary model followed by fallback models if CLI call fails
                models_to_try = self.get_model_candidates(selected_model)

                for attempt_idx, model_candidate in enumerate(models_to_try):
                    output, turn_cid = None, None
                    proc = None
                    try:
                        proc = self._execute_cli(
                            session.conversation_id, cli_prompt, timeout_sec, model_candidate,
                            bot_cli_base, run_env, is_mocked
                        )
                        output, turn_cid = self._parse_cli_payload(proc)
                    except subprocess.TimeoutExpired:
                        if not (session.conversation_id and not conversation_id):
                            if attempt_idx == len(models_to_try) - 1:
                                raise
                            logger.warning(
                                "Model candidate %s timed out after %ss; trying fallback.",
                                model_candidate,
                                timeout_sec,
                            )
                            continue
                        logger.warning(
                            "Resuming topic '%s' conversation %s timed out after %ss; falling back to one-off thread.",
                            topic_key,
                            session.conversation_id,
                            timeout_sec,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Model candidate %s failed with exception: %s; trying fallback.",
                            model_candidate,
                            exc,
                        )
                        continue

                    if output is None and session.conversation_id and not conversation_id:
                        # Runtime resilience: answer this call from a detached one-off thread.
                        logger.info(
                            "Resuming topic '%s' conversation %s failed with model %s; answering from a one-off thread.",
                            topic_key, session.conversation_id, model_candidate,
                        )
                        fallback_prompt = cli_prompt
                        if use_topic_session and topic_key and not session.history_prefix:
                            try:
                                hist_prefix = format_topic_history_prefix(topic_key)
                                if hist_prefix:
                                    fallback_body = f"{hist_prefix}{prompt}"
                                    fallback_prompt = f"{effective_title}\n\n{fallback_body}" if effective_title else fallback_body
                            except Exception:
                                pass
                        try:
                            proc = self._execute_cli(
                                None, fallback_prompt, timeout_sec, model_candidate,
                                bot_cli_base, run_env, is_mocked
                            )
                            output, detached_cid = self._parse_cli_payload(proc)
                            if detached_cid and detached_cid != session.conversation_id and not is_mocked:
                                self._remove_detached_thread(bot_cli_base, detached_cid)
                        except Exception as exc:
                            logger.warning("One-off thread attempt failed for model %s: %s", model_candidate, exc)
                        turn_cid = None

                    if output:
                        if attempt_idx > 0:
                            logger.warning(
                                "LLM query succeeded using fallback model '%s' (primary '%s' failed).",
                                model_candidate,
                                selected_model,
                            )
                        if cache_ttl_sec > 0:
                            self._cache[prompt] = (now, output)
                        if use_topic_session:
                            def _finalize_topic_turn() -> None:
                                compacted_ok = False
                                if session.compact and turn_cid:
                                    compact_timeout = max(timeout_sec, 120.0)
                                    try:
                                        c_proc = self._execute_cli(
                                            turn_cid,
                                            "/compact Summarize prior turns in place, preserving open positions, recent decisions, and active risk guards.",
                                            compact_timeout,
                                            model_candidate,
                                            bot_cli_base,
                                            run_env,
                                            is_mocked,
                                        )
                                        c_out, c_cid = self._parse_cli_payload(c_proc)
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
                            "agy CLI with model '%s' failed (code %s): %s",
                            model_candidate,
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
            rest_candidates: list[str] = []
            for mc in self.get_model_candidates(selected_model):
                rc = canonical_rest_model(mc)
                if rc not in rest_candidates:
                    rest_candidates.append(rc)
            for r_default in ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"):
                if r_default not in rest_candidates:
                    rest_candidates.append(r_default)

            for r_model in rest_candidates:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{r_model}:generateContent?key={self.api_key}"
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
                        candidates = resp_data.get("candidates") if isinstance(resp_data, dict) else None
                        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
                            continue
                        content_obj = candidates[0].get("content")
                        parts = content_obj.get("parts") if isinstance(content_obj, dict) else None
                        if not isinstance(parts, list) or not parts or not isinstance(parts[0], dict):
                            continue
                        raw_text = parts[0].get("text")
                        text = raw_text.strip() if isinstance(raw_text, str) else ""
                        if not text:
                            continue
                        if cache_ttl_sec > 0:
                            self._cache[prompt] = (now, text)
                        return text
                except Exception as e:
                    logger.warning("Gemini REST API attempt with model %s failed: %s", r_model, e)

        return None

    @staticmethod
    def _remove_detached_thread(bot_cli_base: str, cid: str) -> None:
        """Delete a one-off fallback conversation from the isolated bot store so orphans never accumulate."""
        if not bot_cli_base or not isinstance(cid, str):
            return
        cid = cid.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,128}", cid):
            return
        delete_threads({cid}, bot_cli_base)

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
