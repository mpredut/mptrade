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

from botcore import _parse_bool_flag, load_dotenv, required_env, safe_float
from cachepaths import cache_path
from intelligence.macro_sentiment_ssot import is_cache_fresh, safe_pos_float
from orchestratorOS.agy.paths import resolve_agy_bot_bin, resolve_bot_cli_base, resolve_ui_cli_base
from orchestratorOS.agy.summary_store import (
    delete_threads,
    restore_conversation_state,
    snapshot_conversation_state,
)
from orchestratorOS.agy.topics import (
    TopicSession,
    extract_json_object,
    format_topic_history_prefix,
    format_topic_turn_header,
    prepare_topic_session,
    record_topic_turn,
    topic_spec,
)
from state_io import read_json_dict, save_json_state

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT_DIR, "config.env"))

logger = logging.getLogger("intelligence.sentiment.gemini_client")

DEFAULT_MODEL = required_env("LLM_DEFAULT_MODEL")
FALLBACK_MODELS_RAW = required_env("LLM_FALLBACK_MODELS")
DEFAULT_FALLBACK_MODELS = tuple(
    m.strip() for m in FALLBACK_MODELS_RAW.split(",") if m.strip()
)
AGY_CLI_PATH = resolve_agy_bot_bin()
QUOTA_API_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
LOAD_ASSIST_API_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
AGY_QUOTA_USER_AGENT = "antigravity/1.4.12"


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

    _TOPIC_LOCKS: Dict[str, threading.Lock] = {}
    _TOPIC_LOCKS_GUARD = threading.Lock()
    _JSON_MODE_LOCAL = threading.local()
    _QUOTA_LOCK = threading.Lock()
    _QUOTA_MEM_CACHE: Dict[str, Any] = {}

    @classmethod
    def _get_topic_lock(cls, topic_key: str) -> threading.Lock:
        key = str(topic_key or "").strip()
        with cls._TOPIC_LOCKS_GUARD:
            lock = cls._TOPIC_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                cls._TOPIC_LOCKS[key] = lock
            return lock

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cli_path: Optional[str] = None,
        custom_runner: Optional[Callable[[str, str, float], str]] = None,
        project_id: Optional[str] = None,
        fallback_models: Optional[tuple[str, ...] | list[str]] = None,
    ) -> None:
        cleaned_model = model.strip() if isinstance(model, str) and model.strip() else ""
        self._explicit_model = bool(cleaned_model)
        self.model = cleaned_model or DEFAULT_MODEL
        self._custom_fallbacks = fallback_models is not None
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
        if selected.endswith("-low") and not getattr(self, "_custom_fallbacks", False):
            for fb in self.fallback_models:
                if fb and fb not in candidates and (fb.endswith("-low") or fb == "gpt-oss-120b-medium"):
                    candidates.append(fb)
        for fb in self.fallback_models:
            if fb and fb not in candidates:
                candidates.append(fb)
        return candidates

    @staticmethod
    def _get_quota_state_path() -> str:
        """Return shared SSOT cache path for live LLM model quota & credit status."""
        return cache_path("llm_quota_status.json")

    @staticmethod
    def _extract_available_credits(load_assist_payload: Any) -> float:
        """Extract total available paid credits from loadCodeAssist response payload."""
        if not isinstance(load_assist_payload, dict):
            return 0.0
        total = 0.0
        for tier_key in ("paidTier", "currentTier"):
            tier = load_assist_payload.get(tier_key)
            if not isinstance(tier, dict):
                continue
            credits_list = tier.get("availableCredits")
            if not isinstance(credits_list, list):
                continue
            for item in credits_list:
                if isinstance(item, dict):
                    val = safe_float(
                        item.get("creditAmount")
                        or item.get("amount")
                        or item.get("remainingCredits")
                        or item.get("credits"),
                        0.0,
                    )
                    if val > 0:
                        total += val
                elif isinstance(item, (int, float, str)):
                    val = safe_float(item, 0.0)
                    if val > 0:
                        total += val
        return total

    @classmethod
    def probe_live_model_quota(
        cls,
        bot_cli_base: Optional[str] = None,
        *,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Query Cloud Code internal quota & credits endpoints (~200ms) with short TTL cache.

        Ensures a model is only invoked when we know it has remaining subscription quota
        (``remainingFraction > 0``) or paid top-up credits (``available_credits > 0``),
        and dynamically restores primary paid models immediately after a top-up or window reset.
        """
        now = time.time()
        ttl_sec = max(5.0, safe_float(os.environ.get("LLM_QUOTA_CACHE_TTL_SEC"), 30.0))
        state_path = cls._get_quota_state_path()

        with cls._QUOTA_LOCK:
            mem = cls._QUOTA_MEM_CACHE
            if (
                not force_refresh
                and isinstance(mem, dict)
                and isinstance(mem.get("buckets"), dict)
                and (now - safe_float(mem.get("checked_ts"), 0.0)) < ttl_sec
            ):
                return dict(mem)

            disk = read_json_dict(state_path, default={})
            if (
                not force_refresh
                and isinstance(disk, dict)
                and isinstance(disk.get("buckets"), dict)
                and (now - safe_float(disk.get("checked_ts"), 0.0)) < ttl_sec
            ):
                cls._QUOTA_MEM_CACHE = dict(disk)
                return dict(disk)

        cli_bases = []
        if bot_cli_base and isinstance(bot_cli_base, str):
            cli_bases.append(bot_cli_base)
        for resolved_base in (resolve_bot_cli_base(), resolve_ui_cli_base()):
            if resolved_base and resolved_base not in cli_bases:
                cli_bases.append(resolved_base)

        access_token = ""
        for base_dir in cli_bases:
            tok_path = os.path.join(base_dir, "antigravity-oauth-token")
            if not os.path.isfile(tok_path):
                continue
            try:
                with open(tok_path, "r", encoding="utf-8") as fh:
                    tok_data = json.load(fh)
                if not isinstance(tok_data, dict):
                    continue
                nested_tok = tok_data.get("token") if isinstance(tok_data.get("token"), dict) else {}
                tok = str(nested_tok.get("access_token") or tok_data.get("access_token") or "").strip()
                exp_ts = safe_float(
                    nested_tok.get("expiry_timestamp") or tok_data.get("expiry_timestamp"),
                    0.0,
                )
                if tok and (exp_ts <= 0.0 or exp_ts > now + 10.0):
                    access_token = tok
                    break
            except Exception:
                continue

        if not access_token:
            return None

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": AGY_QUOTA_USER_AGENT,
        }

        buckets: Dict[str, float] = {}
        try:
            req = urllib.request.Request(QUOTA_API_URL, data=b"{}", headers=headers)
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                quota_payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(quota_payload, dict) or not isinstance(quota_payload.get("buckets"), list):
                return None
            for b in quota_payload.get("buckets", []):
                if not isinstance(b, dict):
                    continue
                mid = str(b.get("modelId") or "").strip()
                if not mid:
                    continue
                rem = max(0.0, safe_float(b.get("remainingFraction"), 0.0))
                buckets[mid] = rem
            # Cloud Code retrieveUserQuota reports the shared Google Gemini/Argon quota pool
            # under gemini-3.8-flash-tiered / gemini-2.5-pro / gemini-3-flash.
            gemini_pool_rem = max(
                safe_float(buckets.get("gemini-3.8-flash-tiered"), 0.0),
                safe_float(buckets.get("gemini-2.5-pro"), 0.0),
                safe_float(buckets.get("gemini-3-flash"), 0.0),
            )
            if gemini_pool_rem > 0.0:
                for gemini_alias in (
                    "gemini-4-argon-high",
                    "gemini-4-argon-medium",
                    "gemini-4-argon-low",
                    "gemini-3.8-flash-medium",
                    "gemini-3.8-flash-low",
                ):
                    buckets.setdefault(gemini_alias, gemini_pool_rem)
        except Exception as exc:
            logger.debug("Live LLM quota check failed (non-fatal): %s", exc)
            return None

        available_credits = 0.0
        tier_id = ""
        try:
            req_assist = urllib.request.Request(LOAD_ASSIST_API_URL, data=b"{}", headers=headers)
            with urllib.request.urlopen(req_assist, timeout=2.0) as resp_assist:
                assist_payload = json.loads(resp_assist.read().decode("utf-8"))
            if isinstance(assist_payload, dict):
                available_credits = cls._extract_available_credits(assist_payload)
                paid_tier = assist_payload.get("paidTier") if isinstance(assist_payload.get("paidTier"), dict) else {}
                curr_tier = assist_payload.get("currentTier") if isinstance(assist_payload.get("currentTier"), dict) else {}
                tier_id = str(paid_tier.get("id") or curr_tier.get("id") or "").strip()
        except Exception as exc:
            logger.debug("Live LLM loadCodeAssist credit check failed (non-fatal): %s", exc)

        with cls._QUOTA_LOCK:
            prev = read_json_dict(state_path, default={})
            if not isinstance(prev, dict):
                prev = {}
            runtime_exhausted: Dict[str, float] = {}
            prev_exhausted = prev.get("runtime_exhausted")
            if isinstance(prev_exhausted, dict) and available_credits <= 0.0:
                for m_name, exp_val in prev_exhausted.items():
                    exp_ts = safe_float(exp_val, 0.0)
                    if exp_ts > now:
                        runtime_exhausted[str(m_name)] = exp_ts

            snapshot: Dict[str, Any] = {
                "checked_ts": now,
                "tier_id": tier_id,
                "available_credits": available_credits,
                "buckets": buckets,
                "runtime_exhausted": runtime_exhausted,
                "last_alert_ts": safe_float(prev.get("last_alert_ts"), 0.0),
                "last_alert_tier": str(prev.get("last_alert_tier") or "ok"),
            }
            cls._QUOTA_MEM_CACHE = dict(snapshot)
            try:
                save_json_state(state_path, snapshot)
            except Exception:
                pass

        cls._maybe_notify_low_quota(snapshot)
        return snapshot

    @classmethod
    def filter_models_with_confirmed_credits(
        cls,
        candidates: list[str],
        quota_snapshot: Optional[Dict[str, Any]],
    ) -> list[str]:
        """Filter candidate models so only models with confirmed quota or top-up credits are called."""
        if not candidates or not isinstance(quota_snapshot, dict):
            return list(candidates)
        buckets = quota_snapshot.get("buckets")
        if not isinstance(buckets, dict):
            return list(candidates)
        available_credits = safe_float(quota_snapshot.get("available_credits"), 0.0)
        if not buckets and available_credits <= 0.0:
            return list(candidates)

        now = time.time()
        runtime_exhausted = (
            quota_snapshot.get("runtime_exhausted")
            if isinstance(quota_snapshot.get("runtime_exhausted"), dict)
            else {}
        )

        confirmed: list[str] = []
        for m in candidates:
            exp_ts = safe_float(runtime_exhausted.get(m), 0.0)
            if exp_ts > now and available_credits <= 0.0:
                continue
            if available_credits > 0.0 or safe_float(buckets.get(m), 0.0) > 0.0:
                confirmed.append(m)
        return confirmed

    @classmethod
    def _record_model_quota_exhausted(cls, model_name: str, ttl_sec: float = 120.0) -> None:
        """Record a runtime quota/credit error for a model and trigger low-quota alert check."""
        cleaned = str(model_name or "").strip()
        if not cleaned:
            return
        now = time.time()
        state_path = cls._get_quota_state_path()
        snapshot: Optional[Dict[str, Any]] = None
        with cls._QUOTA_LOCK:
            data = read_json_dict(state_path, default={})
            if not isinstance(data, dict):
                data = {}
            buckets = data.get("buckets") if isinstance(data.get("buckets"), dict) else {}
            buckets[cleaned] = 0.0
            runtime_exhausted = (
                data.get("runtime_exhausted")
                if isinstance(data.get("runtime_exhausted"), dict)
                else {}
            )
            runtime_exhausted[cleaned] = now + max(15.0, float(ttl_sec))
            data["buckets"] = buckets
            data["runtime_exhausted"] = runtime_exhausted
            data.setdefault("checked_ts", now)
            data.setdefault("available_credits", 0.0)
            cls._QUOTA_MEM_CACHE = dict(data)
            try:
                save_json_state(state_path, data)
            except Exception:
                pass
            snapshot = dict(data)
        if snapshot is not None:
            cls._maybe_notify_low_quota(snapshot)

    @classmethod
    def _maybe_notify_low_quota(cls, snapshot: Dict[str, Any]) -> None:
        """Send an alert on NTFY_TOPIC_ERROR when primary paid model quota/credits are near zero."""
        if not isinstance(snapshot, dict):
            return
        buckets = snapshot.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            return
        available_credits = safe_float(snapshot.get("available_credits"), 0.0)
        warn_threshold = max(0.0, min(0.5, safe_float(os.environ.get("LLM_QUOTA_LOW_WARN_FRACTION"), 0.10)))
        cooldown_sec = max(300.0, safe_float(os.environ.get("LLM_QUOTA_ALERT_COOLDOWN_SEC"), 14400.0))

        argon_rem = max(
            safe_float(buckets.get("gemini-4-argon-medium"), 0.0),
            safe_float(buckets.get("gemini-4-argon-low"), 0.0),
            safe_float(buckets.get("gemini-4-argon-high"), 0.0),
        )
        claude_rem = max(
            safe_float(buckets.get("claude-sonnet-5-5-medium"), 0.0),
            safe_float(buckets.get("claude-sonnet-5-5-low"), 0.0),
        )
        free_rem = safe_float(buckets.get("gpt-oss-120b-medium"), 0.0)

        if available_credits > 0.0 or argon_rem > warn_threshold:
            if snapshot.get("last_alert_tier") not in (None, "", "ok"):
                state_path = cls._get_quota_state_path()
                with cls._QUOTA_LOCK:
                    snapshot["last_alert_tier"] = "ok"
                    cls._QUOTA_MEM_CACHE = dict(snapshot)
                    try:
                        save_json_state(state_path, snapshot)
                    except Exception:
                        pass
            return

        if claude_rem <= warn_threshold and free_rem <= warn_threshold:
            alert_tier = "all_exhausted"
            severity_rank = 3
        elif claude_rem <= warn_threshold:
            alert_tier = "all_paid_low"
            severity_rank = 2
        else:
            alert_tier = "argon_low"
            severity_rank = 1

        rank_map = {"ok": 0, "argon_low": 1, "all_paid_low": 2, "all_exhausted": 3}
        prev_tier = str(snapshot.get("last_alert_tier") or "ok")
        prev_rank = rank_map.get(prev_tier, 0)
        now = time.time()
        last_alert_ts = safe_float(snapshot.get("last_alert_ts"), 0.0)

        if severity_rank <= prev_rank and (now - last_alert_ts) < cooldown_sec:
            return

        try:
            from orchestratorOS.lib import watchdog_common
        except Exception:
            return

        if (
            "PYTEST_CURRENT_TEST" in os.environ
            and not hasattr(watchdog_common.send_ntfy, "_mock_name")
            and not _parse_bool_flag(os.environ.get("LLM_FORCE_QUOTA_ALERT_TEST"))
        ):
            return

        title = (
            "🛑 ERROR: AI LLM Quota Near Zero"
            if argon_rem > 0.0
            else "🛑 ERROR: AI LLM Paid Argon Quota Exhausted (0%)"
        )
        msg = (
            f"Credite suplimentare: {available_credits:.0f} | "
            f"gemini-4-argon (med/low/high): {argon_rem * 100:.0f}% | "
            f"claude-sonnet-5-5 (paid fallback): {claude_rem * 100:.0f}% | "
            f"gpt-oss-120b (free): {free_rem * 100:.0f}%. "
            f"Routing activat automat pe primul model cu credite confirmate; "
            f"revenire dinamica imediata pe Argon dupa reincarcare/reset."
        )
        dispatched = False
        try:
            watchdog_common.send_ntfy(title, msg)
            dispatched = True
        except Exception as exc:
            logger.debug("Failed sending low-quota NTFY alert: %s", exc)

        if dispatched:
            state_path = cls._get_quota_state_path()
            with cls._QUOTA_LOCK:
                snapshot["last_alert_ts"] = now
                snapshot["last_alert_tier"] = alert_tier
                cls._QUOTA_MEM_CACHE = dict(snapshot)
                try:
                    save_json_state(state_path, snapshot)
                except Exception:
                    pass

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

    @staticmethod
    def _is_quota_or_credit_error(proc: Optional[subprocess.CompletedProcess]) -> bool:
        """Return True when the CLI failed due to exhausted AI credits, quota, rate limits, or unrecognized model."""
        if proc is None:
            return False
        combined = f"{proc.stderr or ''}\n{proc.stdout or ''}".lower()
        return any(
            tok in combined
            for tok in (
                "credits balance is too low",
                "resource_exhausted",
                "quota exceeded",
                "check quota",
                "rate limit",
                "too many requests",
                "invalid model selection",
                "not recognized as a known model",
                "unknown model",
                "model not found",
            )
        )

    @staticmethod
    def _list_conversation_cids(bot_cli_base: str) -> set[str]:
        """Return existing conversation IDs in ``<bot_cli_base>/conversations``."""
        if not bot_cli_base or not isinstance(bot_cli_base, str):
            return set()
        conv_dir = os.path.join(bot_cli_base, "conversations")
        try:
            return {
                fn[:-3]
                for fn in os.listdir(conv_dir)
                if fn.endswith(".db") and len(fn) > 3
            }
        except OSError:
            return set()

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
        explicit_call_model = model.strip() if isinstance(model, str) and model.strip() else None
        topic_preferred_model: Optional[str] = None
        if not explicit_call_model and not getattr(self, "_explicit_model", False) and topic_key:
            try:
                topic_preferred_model = topic_spec(topic_key).preferred_model
            except Exception:
                topic_preferred_model = None
        selected_model = explicit_call_model or topic_preferred_model or self.model
        timeout_sec = safe_pos_float(timeout_sec, 15.0)
        cache_ttl_sec = safe_pos_float(cache_ttl_sec, 0.0)
        require_json = bool(getattr(self._JSON_MODE_LOCAL, "require_json", False))

        if cache_ttl_sec > 0:
            cached = self._cache.get(prompt)
            if isinstance(cached, (tuple, list)) and len(cached) >= 2 and is_cache_fresh(cached[0], now_ts=now, ttl_sec=cache_ttl_sec):
                if not require_json or extract_json_object(str(cached[1])) is not None:
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
            topic_lock: Optional[threading.Lock] = None
            topic_lock_transferred = False
            try:
                bot_cli_base = resolve_bot_cli_base()
                is_mocked = hasattr(subprocess.run, "_mock_name")
                use_topic_session = bool(
                    topic_key and (not is_mocked or hasattr(prepare_topic_session, "_mock_name"))
                )
                if use_topic_session and topic_key:
                    topic_lock = self._get_topic_lock(topic_key)
                    topic_lock.acquire()
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
                raw_candidates = self.get_model_candidates(selected_model)
                enable_pre_quota = not is_mocked or hasattr(self.probe_live_model_quota, "_mock_name")
                if enable_pre_quota:
                    quota_snapshot = self.probe_live_model_quota(bot_cli_base=bot_cli_base)
                    models_to_try = self.filter_models_with_confirmed_credits(raw_candidates, quota_snapshot)
                else:
                    models_to_try = raw_candidates
                conv_snapshot = (
                    snapshot_conversation_state(bot_cli_base, session.conversation_id)
                    if (session.conversation_id and not is_mocked)
                    else None
                )

                for attempt_idx, model_candidate in enumerate(models_to_try):
                    output, turn_cid = None, None
                    replaced_conversation = False
                    proc = None
                    pre_attempt_cids = (
                        self._list_conversation_cids(bot_cli_base)
                        if (not session.conversation_id and not is_mocked)
                        else set()
                    )
                    try:
                        proc = self._execute_cli(
                            session.conversation_id, cli_prompt, timeout_sec, model_candidate,
                            bot_cli_base, run_env, is_mocked
                        )
                        output, turn_cid = self._parse_cli_payload(proc)
                    except subprocess.TimeoutExpired:
                        if not is_mocked:
                            if session.conversation_id and conv_snapshot is not None:
                                restore_conversation_state(bot_cli_base, session.conversation_id, conv_snapshot)
                            elif not session.conversation_id:
                                for orphan_cid in self._list_conversation_cids(bot_cli_base) - pre_attempt_cids:
                                    self._remove_detached_thread(bot_cli_base, orphan_cid)
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
                        if not is_mocked:
                            if session.conversation_id and conv_snapshot is not None:
                                restore_conversation_state(bot_cli_base, session.conversation_id, conv_snapshot)
                            elif not session.conversation_id:
                                for orphan_cid in self._list_conversation_cids(bot_cli_base) - pre_attempt_cids:
                                    self._remove_detached_thread(bot_cli_base, orphan_cid)
                        logger.warning(
                            "Model candidate %s failed with exception: %s; trying fallback.",
                            model_candidate,
                            exc,
                        )
                        continue

                    if require_json and isinstance(output, str) and extract_json_object(output) is None:
                        logger.warning(
                            "agy CLI with model '%s' returned non-JSON output (%s); rolling back and trying fallback.",
                            model_candidate,
                            output.strip()[:160],
                        )
                        if not is_mocked:
                            if session.conversation_id and conv_snapshot is not None:
                                restore_conversation_state(bot_cli_base, session.conversation_id, conv_snapshot)
                            elif not session.conversation_id:
                                for orphan_cid in self._list_conversation_cids(bot_cli_base) - pre_attempt_cids:
                                    self._remove_detached_thread(bot_cli_base, orphan_cid)
                        output = None
                        turn_cid = None
                        continue

                    if output is None and not is_mocked:
                        if session.conversation_id and conv_snapshot is not None:
                            restore_conversation_state(bot_cli_base, session.conversation_id, conv_snapshot)
                        elif not session.conversation_id:
                            for orphan_cid in self._list_conversation_cids(bot_cli_base) - pre_attempt_cids:
                                self._remove_detached_thread(bot_cli_base, orphan_cid)

                    if (
                        output is None
                        and session.conversation_id
                        and not conversation_id
                        and not self._is_quota_or_credit_error(proc)
                    ):
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
                        pre_cids = self._list_conversation_cids(bot_cli_base) if not is_mocked else set()
                        detached_cid = None
                        try:
                            proc = self._execute_cli(
                                None, fallback_prompt, timeout_sec, model_candidate,
                                bot_cli_base, run_env, is_mocked
                            )
                            output, detached_cid = self._parse_cli_payload(proc)
                            if require_json and isinstance(output, str) and extract_json_object(output) is None:
                                logger.warning(
                                    "agy CLI one-off thread with model '%s' returned non-JSON output (%s); trying fallback.",
                                    model_candidate,
                                    output.strip()[:160],
                                )
                                output = None
                            if (
                                detached_cid
                                and detached_cid != session.conversation_id
                                and not is_mocked
                                and not (output and use_topic_session)
                            ):
                                self._remove_detached_thread(bot_cli_base, detached_cid)
                        except Exception as exc:
                            logger.warning("One-off thread attempt failed for model %s: %s", model_candidate, exc)
                        finally:
                            if not is_mocked and not detached_cid:
                                for orphan_cid in self._list_conversation_cids(bot_cli_base) - pre_cids:
                                    if orphan_cid != session.conversation_id:
                                        self._remove_detached_thread(bot_cli_base, orphan_cid)
                        if output and use_topic_session and detached_cid:
                            turn_cid = detached_cid
                            replaced_conversation = detached_cid != session.conversation_id
                        else:
                            turn_cid = None

                    if output:
                        if attempt_idx > 0 or model_candidate != selected_model:
                            logger.warning(
                                "LLM query succeeded using fallback model '%s' (primary '%s' unavailable/failed).",
                                model_candidate,
                                selected_model,
                            )
                        if cache_ttl_sec > 0:
                            self._cache[prompt] = (now, output)
                        if use_topic_session:
                            def _finalize_topic_turn(release_lock: bool = False) -> None:
                                try:
                                    compacted_ok = False
                                    if session.compact and turn_cid and not replaced_conversation:
                                        compact_timeout = max(timeout_sec, 120.0)
                                        compact_snap = (
                                            snapshot_conversation_state(bot_cli_base, turn_cid)
                                            if not is_mocked
                                            else None
                                        )
                                        for compact_model in models_to_try[attempt_idx:]:
                                            try:
                                                c_proc = self._execute_cli(
                                                    turn_cid,
                                                    "/compact Summarize prior turns in place, preserving open positions, recent decisions, and active risk guards.",
                                                    compact_timeout,
                                                    compact_model,
                                                    bot_cli_base,
                                                    run_env,
                                                    is_mocked,
                                                )
                                                c_out, c_cid = self._parse_cli_payload(c_proc)
                                                compacted_ok = bool(c_out or c_cid)
                                                if compacted_ok:
                                                    break
                                                if not is_mocked and compact_snap is not None:
                                                    restore_conversation_state(bot_cli_base, turn_cid, compact_snap)
                                            except Exception as exc:
                                                if not is_mocked and compact_snap is not None:
                                                    restore_conversation_state(bot_cli_base, turn_cid, compact_snap)
                                                logger.warning(
                                                    "In-place /compact for topic '%s' conversation %s with model %s failed: %s",
                                                    topic_key,
                                                    turn_cid,
                                                    compact_model,
                                                    exc,
                                                )
                                                break
                                    try:
                                        record_kwargs: Dict[str, Any] = {
                                            "topic_key": topic_key,
                                            "conversation_id": turn_cid,
                                            "response_text": output,
                                            "compacted": compacted_ok,
                                            "cli_base": bot_cli_base,
                                        }
                                        if replaced_conversation:
                                            record_kwargs["replace_conversation"] = True
                                        record_topic_turn(**record_kwargs)
                                    except Exception as e:
                                        logger.debug("Failed recording topic turn for %s: %s", topic_key, e)
                                finally:
                                    if release_lock and topic_lock is not None:
                                        topic_lock.release()

                            if session.compact and turn_cid and not is_mocked:
                                topic_lock_transferred = True
                                threading.Thread(target=_finalize_topic_turn, args=(True,), daemon=False).start()
                            else:
                                _finalize_topic_turn(False)
                        return output
                    else:
                        if enable_pre_quota and self._is_quota_or_credit_error(proc):
                            self._record_model_quota_exhausted(model_candidate)
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
            finally:
                if topic_lock is not None and not topic_lock_transferred:
                    topic_lock.release()

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
                        if require_json and extract_json_object(text) is None:
                            logger.warning(
                                "Gemini REST API with model %s returned non-JSON output; trying next.",
                                r_model,
                            )
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
        self._JSON_MODE_LOCAL.require_json = True
        try:
            raw_text = self.query_text(
                prompt,
                model=model,
                timeout_sec=timeout_sec,
                cache_ttl_sec=cache_ttl_sec,
                thread_title=thread_title,
                topic_key=topic_key,
                conversation_id=conversation_id,
            )
        finally:
            self._JSON_MODE_LOCAL.require_json = False
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
