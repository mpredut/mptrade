"""Google Gemini client providing LLM reasoning via authenticated CLI or REST API."""
from __future__ import annotations

import importlib
import logging
import os
import re

_DETACHED_CID_RE = re.compile(r"[A-Za-z0-9_-]{4,128}")
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import urllib.request

from botcore import (
    FileLock,
    clone_process_env,
    fetch_json_post,
    float_env,
    int_env,
    is_cache_fresh,
    is_gemini_llm_disabled,
    is_unmocked_llm_query_allowed,
    load_dotenv,
    optional_env,
    required_env,
    resolve_llm_soft_extend_mult,
    resolve_llm_timeout_sec,
    safe_float,
    safe_pos_float,
    safe_pos_int,
    split_csv,
    topic_spec,
)
from cachepaths import cache_path  # noqa: F401
from intelligence.sentiment.llm_quota_router import (
    LOAD_ASSIST_API_URL,  # noqa: F401
    QUOTA_API_URL,  # noqa: F401
    GeminiQuotaRouterMixin,
    canonical_rest_model,
    is_degraded_trading_model,
)
from intelligence.sentiment.openai_compat_rest import (
    query_openai_compat_chat,
    resolve_openai_compat_config,
    strip_openai_compat_prefix,
)
from state_io import (
    extract_json_object,
    parse_json_payload,
    read_json_dict,  # noqa: F401
    save_json_state,  # noqa: F401
)

_agy_paths = importlib.import_module("orchestratorOS.agy.paths")
resolve_agy_bot_bin = _agy_paths.resolve_agy_bot_bin
resolve_bot_cli_base = _agy_paths.resolve_bot_cli_base

_agy_summary_store = importlib.import_module("orchestratorOS.agy.summary_store")
delete_threads = _agy_summary_store.delete_threads
restore_conversation_state = _agy_summary_store.restore_conversation_state
snapshot_conversation_state = _agy_summary_store.snapshot_conversation_state

_agy_topics = importlib.import_module("orchestratorOS.agy.topics")
TopicSession = _agy_topics.TopicSession
condense_decision = _agy_topics.condense_decision
format_topic_history_prefix = _agy_topics.format_topic_history_prefix
format_topic_turn_header = _agy_topics.format_topic_turn_header
get_topic_slow_rotation_count = _agy_topics.get_topic_slow_rotation_count
is_topic_resume_warm = _agy_topics.is_topic_resume_warm
prepare_topic_session = _agy_topics.prepare_topic_session
record_topic_resume_timeout = _agy_topics.record_topic_resume_timeout
record_topic_slow_rotation = _agy_topics.record_topic_slow_rotation
record_topic_turn = _agy_topics.record_topic_turn

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

logger = logging.getLogger("intelligence.sentiment.gemini_client")

DEFAULT_MODEL = required_env("LLM_DEFAULT_MODEL")
FALLBACK_MODELS_RAW = required_env("LLM_FALLBACK_MODELS")
DEFAULT_FALLBACK_MODELS = split_csv(FALLBACK_MODELS_RAW)
AGY_CLI_PATH = resolve_agy_bot_bin()

_ORIG_SUBPROCESS_RUN = subprocess.run
_ORIG_URLOPEN = urllib.request.urlopen
_ORIG_FETCH_JSON_POST = fetch_json_post
_ORIG_PREPARE_TOPIC_SESSION = prepare_topic_session
_ORIG_RECORD_TOPIC_RESUME_TIMEOUT = record_topic_resume_timeout
_ORIG_RECORD_TOPIC_SLOW_ROTATION = record_topic_slow_rotation
_ORIG_PROBE_LIVE_MODEL_QUOTA_FUNC = getattr(
    GeminiQuotaRouterMixin.probe_live_model_quota, "__func__", GeminiQuotaRouterMixin.probe_live_model_quota
)


class GeminiClient(GeminiQuotaRouterMixin):
    """Wrapper for querying LLM models.

    Prefers the local authenticated `agy` CLI (leveraging active user subscription)
    with fallback to Google AI Studio REST API if an API key is configured.
    """

    _TOPIC_LOCKS: Dict[str, threading.Lock] = {}
    _TOPIC_LOCKS_GUARD = threading.Lock()
    _JSON_MODE_LOCAL = threading.local()

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cli_path: Optional[str] = None,
        custom_runner: Optional[Callable[[str, str, float], str]] = None,
        project_id: Optional[str] = None,
        fallback_models: Optional[tuple[str, ...] | list[str]] = None,
        openai_compat_url: Optional[str] = None,
        openai_compat_api_key: Optional[str] = None,
        openai_compat_models: Optional[tuple[str, ...] | list[str]] = None,
    ) -> None:
        cleaned_model = model.strip() if isinstance(model, str) and model.strip() else ""
        self._explicit_model = bool(cleaned_model)
        self.model = cleaned_model or (optional_env("LLM_DEFAULT_MODEL", "") or "").strip() or DEFAULT_MODEL
        self._custom_fallbacks = fallback_models is not None
        if fallback_models is not None:
            self.fallback_models = tuple(
                m.strip() for m in fallback_models if isinstance(m, str) and m.strip()
            )
        else:
            env_fb = split_csv(optional_env("LLM_FALLBACK_MODELS", ""))
            self.fallback_models = env_fb if env_fb else DEFAULT_FALLBACK_MODELS
        raw_key = (
            api_key
            if isinstance(api_key, str) and api_key.strip()
            else (optional_env("GEMINI_API_KEY") or optional_env("GOOGLE_API_KEY"))
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
        self._openai_compat_url = openai_compat_url
        self._openai_compat_api_key = openai_compat_api_key
        self._openai_compat_models = openai_compat_models
        self._cache: Dict[Any, tuple[Any, ...]] = {}
        self.last_model_used: Optional[str] = None

    def _cache_get(
        self,
        selected_model: str,
        prompt: str,
        *,
        now: float,
        cache_ttl_sec: float,
        require_json: bool,
        allow_legacy_key: bool = True,
    ) -> Optional[str]:
        if cache_ttl_sec <= 0:
            return None
        cached = self._cache.get((selected_model, prompt))
        if cached is None and allow_legacy_key:
            cached = self._cache.get(prompt)
        if (
            isinstance(cached, (tuple, list))
            and len(cached) >= 2
            and is_cache_fresh(cached[0], now_ts=now, ttl_sec=cache_ttl_sec)
        ):
            if not require_json or extract_json_object(str(cached[1])) is not None:
                self.last_model_used = str(cached[2]) if len(cached) >= 3 and cached[2] else selected_model
                return cached[1]
        return None

    def _cache_put(
        self,
        selected_model: str,
        prompt: str,
        *,
        now: float,
        text: str,
        model_used: Optional[str],
        cache_ttl_sec: float,
    ) -> None:
        if cache_ttl_sec > 0 and text:
            self._cache[(selected_model, prompt)] = (now, text, model_used or selected_model)

    @staticmethod
    def _run_subprocess_with_soft_extend(
        cmd: list[str],
        *,
        soft_timeout: float,
        extend_mult: float,
        run_env: dict[str, str],
        exec_cwd: Optional[str],
        cli_model: str,
        resume: Optional[str],
    ) -> tuple[subprocess.CompletedProcess, bool]:
        """Run ``cmd`` via ``Popen``: on soft timeout do NOT kill -- extend up to ``soft_timeout * extend_mult``."""
        hard_timeout = max(soft_timeout, soft_timeout * max(1.0, extend_mult))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=run_env,
            cwd=exec_cwd,
        )
        exceeded_soft = False
        reaped = False
        try:
            try:
                stdout, stderr = proc.communicate(timeout=soft_timeout)
            except subprocess.TimeoutExpired:
                exceeded_soft = True
                extra_wait = max(1.0, hard_timeout - soft_timeout)
                if hard_timeout <= soft_timeout + 1e-6:
                    proc.kill()
                    proc.communicate()
                    reaped = True
                    raise
                logger.warning(
                    "agy CLI (model=%s, conv=%s) exceeded soft timeout %.0fs; NOT killing -- extending wait up to %.0fs so execution finishes cleanly and scheduling thread rotation for next turn.",
                    cli_model,
                    resume or "new",
                    soft_timeout,
                    hard_timeout,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=extra_wait)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    reaped = True
                    raise subprocess.TimeoutExpired(cmd, hard_timeout)
        except BaseException:
            if not reaped:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    if hasattr(proc, "wait"):
                        proc.wait(timeout=5.0)
                    else:
                        proc.communicate()
                except Exception:
                    pass
            raise
        return (
            subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode if proc.returncode is not None else 0,
                stdout=stdout,
                stderr=stderr,
            ),
            exceeded_soft,
        )

    def _execute_cli(
        self,
        resume: Optional[str],
        text: str,
        call_timeout: float,
        cli_model: str,
        bot_cli_base: str,
        run_env: dict[str, str],
        is_mocked: bool,
        *,
        include_repo_rules: Optional[bool] = None,
        allow_soft_extend: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run the isolated agy CLI tool with robust file-locking and soft-timeout extension.

        Non-coding autonomous queries (trading, macro, sentiment, pre-trade guards) run with
        ``include_repo_rules=False`` from ``bot_cli_base`` so ``agy`` does not walk up into
        ``_ROOT_DIR`` or inject ``AGENTS.md`` engineering rules into financial reasoning prompts.
        Code self-healing topics (``can_self_heal=True``) set ``include_repo_rules=True`` to run
        inside ``_ROOT_DIR`` with ``--add-dir`` and load ``AGENTS.md``.
        """
        self._JSON_MODE_LOCAL.exceeded_soft_timeout = False
        use_repo_rules = (
            bool(getattr(self._JSON_MODE_LOCAL, "include_repo_rules", False))
            if include_repo_rules is None
            else bool(include_repo_rules)
        )
        cmd = [self.cli_path, "--project", self.project_id, "--model", cli_model, "--output-format", "json"]
        if not is_mocked and use_repo_rules and os.path.isdir(_ROOT_DIR):
            cmd.extend(["--add-dir", _ROOT_DIR])
        if resume:
            cmd.extend(["--conversation", resume])
        cmd.extend(["-p", text])
        if is_mocked or not os.path.isdir(bot_cli_base):
            return subprocess.run(cmd, capture_output=True, text=True, timeout=call_timeout,
                                  check=False, env=run_env)
        exec_cwd = _ROOT_DIR if (use_repo_rules and os.path.isdir(_ROOT_DIR)) else bot_cli_base
        extend_mult = resolve_llm_soft_extend_mult(3.0) if allow_soft_extend else 1.0
        effective_max_wait = call_timeout * extend_mult
        lock_path = os.path.join(bot_cli_base, ".agy_bot_cli.lock")
        lock_wait = max(90.0, effective_max_wait) if call_timeout >= 30.0 else min(call_timeout * 0.5, 5.0)
        with FileLock(lock_path, timeout=lock_wait, poll_interval=0.15, raise_on_timeout=False):
            pre_dirty = (
                self._snapshot_dirty_tracked_files(_ROOT_DIR)
                if (use_repo_rules and os.path.isdir(_ROOT_DIR))
                else None
            )
            try:
                if allow_soft_extend and extend_mult > 1.0:
                    completed, exceeded_soft = self._run_subprocess_with_soft_extend(
                        cmd,
                        soft_timeout=call_timeout,
                        extend_mult=extend_mult,
                        run_env=run_env,
                        exec_cwd=exec_cwd,
                        cli_model=cli_model,
                        resume=resume,
                    )
                    self._JSON_MODE_LOCAL.exceeded_soft_timeout = exceeded_soft
                    return completed
                return subprocess.run(cmd, capture_output=True, text=True, timeout=call_timeout,
                                      check=False, env=run_env, cwd=exec_cwd)
            finally:
                if pre_dirty is not None:
                    self._restore_newly_dirtied_tracked_files(_ROOT_DIR, pre_dirty)

    @staticmethod
    def _snapshot_dirty_tracked_files(repo_dir: str) -> Optional[set[str]]:
        """Return relative paths of tracked files currently modified vs HEAD in ``repo_dir``."""
        if not repo_dir or not os.path.isdir(repo_dir):
            return None
        default_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not is_unmocked_llm_query_allowed() and os.path.abspath(repo_dir) == default_root:
            return None
        try:
            proc = _ORIG_SUBPROCESS_RUN(
                ["git", "-C", repo_dir, "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if proc.returncode != 0:
                return None
            return {line.strip() for line in (proc.stdout or "").splitlines() if line.strip()}
        except Exception:
            return None

    @classmethod
    def _restore_newly_dirtied_tracked_files(cls, repo_dir: str, pre_dirty: Optional[set[str]]) -> None:
        """Restore any tracked files in ``repo_dir`` that were clean before an agy CLI query/compact call."""
        if pre_dirty is None:
            return
        post_dirty = cls._snapshot_dirty_tracked_files(repo_dir)
        if not post_dirty:
            return
        newly_dirty = sorted(post_dirty - pre_dirty)
        if not newly_dirty:
            return
        logger.warning(
            "Restoring %d tracked workspace file(s) modified directly during agy CLI call: %s",
            len(newly_dirty),
            ", ".join(newly_dirty),
        )
        try:
            proc = _ORIG_SUBPROCESS_RUN(
                ["git", "-C", repo_dir, "checkout", "HEAD", "--", *newly_dirty],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if getattr(proc, "returncode", 0) != 0 and len(newly_dirty) >= 1:
                for rel_p in newly_dirty:
                    p_one = _ORIG_SUBPROCESS_RUN(
                        ["git", "-C", repo_dir, "checkout", "HEAD", "--", rel_p],
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                        check=False,
                    )
                    if getattr(p_one, "returncode", 0) != 0:
                        _ORIG_SUBPROCESS_RUN(
                            ["git", "-C", repo_dir, "reset", "HEAD", "--", rel_p],
                            capture_output=True,
                            text=True,
                            timeout=5.0,
                            check=False,
                        )
        except Exception as exc:
            logger.warning("Failed restoring newly dirtied tracked files %s: %s", newly_dirty, exc)

    @staticmethod
    def _parse_cli_payload(proc: subprocess.CompletedProcess) -> Tuple[Optional[str], Optional[str]]:
        """Extract response text and conversation_id from agy CLI JSON output."""
        raw = (proc.stdout or "").strip()
        if proc.returncode != 0 or not raw:
            return None, None
        payload = parse_json_payload(raw, default=None)
        if not isinstance(payload, dict):
            return raw, None
        raw_resp = payload.get("response")
        text_out = raw_resp.strip() if isinstance(raw_resp, str) else ""
        raw_cid = payload.get("conversation_id")
        cid_out = raw_cid.strip() if isinstance(raw_cid, str) and raw_cid.strip() else None
        return (text_out or None), cid_out

    @staticmethod
    def _is_cli_interrupted(proc: Optional[subprocess.CompletedProcess]) -> bool:
        """Return True when the CLI subprocess was interrupted by a signal (e.g. SIGTERM/SIGINT)."""
        if proc is None:
            return False
        rc = getattr(proc, "returncode", 0) or 0
        if rc < 0 or rc in (130, 143):
            return True
        err_text = (proc.stderr or "").strip().lower()
        return any(
            tok in err_text
            for tok in (
                "error: interrupted",
                "context canceled",
                "signal: terminated",
                "signal: interrupt",
                "terminated by signal",
            )
        )

    _UNRECOGNIZED_MODEL_TOKENS = (
        "invalid model selection",
        "not recognized as a known model",
        "unknown model",
        "model not found",
    )

    @classmethod
    def _cli_error_text(cls, proc: Optional[subprocess.CompletedProcess]) -> str:
        if proc is None:
            return ""
        err_raw = (proc.stderr or "").strip()
        out_raw = (proc.stdout or "").strip()
        # Inspect stderr when present; only fall back to stdout when stderr is empty
        # so buffered prompt/response text on stdout (e.g. "rate limit") is never mistaken for a CLI error.
        return (err_raw if err_raw else out_raw).lower()

    @classmethod
    def _is_unrecognized_model_error(cls, proc: Optional[subprocess.CompletedProcess]) -> bool:
        """Return True when the CLI rejected the requested model slug as unknown or unavailable on the active tier."""
        combined = "" if (proc is None or cls._is_cli_interrupted(proc)) else cls._cli_error_text(proc)
        return bool(combined) and any(tok in combined for tok in cls._UNRECOGNIZED_MODEL_TOKENS)

    @classmethod
    def _is_quota_or_credit_error(cls, proc: Optional[subprocess.CompletedProcess]) -> bool:
        """Return True when the CLI failed due to exhausted AI credits, quota, rate limits, or unrecognized model."""
        if cls._is_unrecognized_model_error(proc):
            return True
        combined = "" if (proc is None or cls._is_cli_interrupted(proc)) else cls._cli_error_text(proc)
        return bool(combined) and any(
            tok in combined
            for tok in (
                "credits balance is too low",
                "resource_exhausted",
                "quota exceeded",
                "check quota",
                "rate limit",
                "too many requests",
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

    def _attempt_detached_one_off_thread(
        self,
        *,
        topic_key: Optional[str],
        session: TopicSession,
        model_candidate: str,
        models_to_try: list[str],
        attempt_idx: int,
        resume_timed_out: bool,
        timeout_sec: float,
        fallback_prompt: str,
        bot_cli_base: str,
        run_env: dict[str, str],
        is_mocked: bool,
        require_json: bool,
        use_topic_session: bool,
        tried_detached_models: set[str],
    ) -> tuple[Optional[str], Optional[str], str, bool]:
        """Fallback to a detached one-off thread when resuming a topic conversation fails."""
        one_off_model = (
            models_to_try[attempt_idx + 1]
            if (
                resume_timed_out
                and any(tag in model_candidate for tag in ("-high", "-xhigh"))
                and attempt_idx + 1 < len(models_to_try)
            )
            else model_candidate
        )
        # Derive one-off fallback cap as 5/8 of primary budget (e.g. 75s @ 120s, 150s @ 240s)
        one_off_cap = timeout_sec * 0.625
        one_off_timeout = one_off_cap if resume_timed_out else timeout_sec
        tried_detached_models.add(one_off_model)
        logger.info(
            "Resuming topic '%s' conversation %s failed with model %s; answering from a one-off thread (model=%s, timeout=%ss).",
            topic_key, session.conversation_id, model_candidate, one_off_model, one_off_timeout,
        )
        pre_cids = self._list_conversation_cids(bot_cli_base) if not is_mocked else set()
        output, detached_cid = None, None
        try:
            proc = self._execute_cli(
                None, fallback_prompt, one_off_timeout, one_off_model,
                bot_cli_base, run_env, is_mocked
            )
            output, detached_cid = self._parse_cli_payload(proc)
            if require_json and isinstance(output, str) and extract_json_object(output) is None:
                logger.warning(
                    "agy CLI one-off thread with model '%s' returned non-JSON output (%s); trying fallback.",
                    one_off_model,
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
        except subprocess.TimeoutExpired:
            logger.warning(
                "One-off thread attempt timed out after %ss for model %s.",
                one_off_timeout,
                one_off_model,
            )
        except Exception as exc:
            logger.warning("One-off thread attempt failed for model %s: %s", one_off_model, exc)
        finally:
            if not is_mocked and not detached_cid:
                self._cleanup_orphan_threads(
                    bot_cli_base, pre_cids, keep_cid=session.conversation_id
                )
        effective_model_used = one_off_model if output else model_candidate
        if output and use_topic_session and detached_cid:
            return output, detached_cid, effective_model_used, (detached_cid != session.conversation_id)
        return output, None, effective_model_used, False

    def _maybe_step_down_after_slow_rotations(
        self,
        *,
        topic_key: Optional[str],
        slow_model: str,
        models_to_try: list[str],
        slow_count: int,
    ) -> None:
        """After N consecutive slow rotations on ``slow_model``, step down to the next faster non-degraded model and alert."""
        if not topic_key or not slow_model:
            return
        threshold = safe_pos_int(int_env("LLM_SLOW_ROTATIONS_FOR_FALLBACK", default=2), 2) or 2
        if slow_count < threshold:
            return
        try:
            can_trade = bool(getattr(topic_spec(topic_key), "can_trade", False))
        except Exception:
            can_trade = False
        slow_lc = slow_model.strip().lower()
        search_pool = list(dict.fromkeys([*models_to_try, *self.get_model_candidates(slow_model)]))
        faster_model = ""
        passed_slow = False
        for cand in search_pool:
            c_clean = cand.strip() if isinstance(cand, str) else ""
            if not c_clean:
                continue
            if c_clean.lower() == slow_lc:
                passed_slow = True
                continue
            if not passed_slow:
                continue
            if can_trade and is_degraded_trading_model(c_clean):
                continue
            faster_model = c_clean
            break
        if faster_model:
            self.record_model_latency_demotion(
                slow_model,
                faster_model,
                topic_key=topic_key,
                slow_rotations=slow_count,
                notify=True,
            )

    def _finalize_topic_turn(
        self,
        *,
        topic_key: Optional[str],
        session: TopicSession,
        turn_cid: Optional[str],
        output: str,
        replaced_conversation: bool,
        timeout_sec: float,
        models_to_try: list[str],
        attempt_idx: int,
        enable_pre_quota: bool,
        bot_cli_base: str,
        run_env: dict[str, str],
        is_mocked: bool,
        include_repo_rules: bool,
        topic_lock: Optional[threading.Lock],
        release_lock: bool = False,
        schedule_rotation: bool = False,
        count_slow_rotation: bool = False,
        timed_out_model: str = "",
        effective_model_used: str = "",
    ) -> None:
        """Optionally run in-place /compact on a completed topic turn and record turn metadata."""
        prev_finalize_rules = getattr(self._JSON_MODE_LOCAL, "include_repo_rules", False)
        self._JSON_MODE_LOCAL.include_repo_rules = include_repo_rules
        try:
            compacted_ok = False
            compact_slow = False
            main_turn_slow = bool(schedule_rotation or count_slow_rotation)
            if session.compact and turn_cid and not replaced_conversation and not schedule_rotation:
                compact_timeout = max(timeout_sec, resolve_llm_timeout_sec())
                compact_snap = (
                    snapshot_conversation_state(bot_cli_base, turn_cid)
                    if not is_mocked
                    else None
                )
                latest_turn_summary = condense_decision(output, max_len=360)
                compact_prompt = (
                    f"/compact Context checkpoint after completed turn ({latest_turn_summary}). "
                    "All prior trading turns (including single-entry HOLD JSON responses) are ALREADY COMPLETED and finalized; "
                    "this NEW /compact request has NOT been answered yet, so in the checkpoint 'Next Steps' section write ONLY: "
                    "'Reply ONLY with COMPACT_OK and wait for the next user prompt.' "
                    "In your immediate post-compaction response, reply ONLY with the single token COMPACT_OK "
                    "(do NOT output any market analysis, markdown report, JSON blocks, position calculations, "
                    "stop-loss/risk rules, or tool calls)."
                )
                for compact_model in self.get_compact_model_candidates(
                    models_to_try,
                    attempt_idx,
                    quota_snapshot=self._QUOTA_MEM_CACHE if enable_pre_quota else None,
                ):
                    try:
                        c_exec_kwargs: Dict[str, Any] = {}
                        if not is_mocked:
                            c_exec_kwargs["allow_soft_extend"] = True
                        c_proc = self._execute_cli(
                            turn_cid,
                            compact_prompt,
                            compact_timeout,
                            compact_model,
                            bot_cli_base,
                            run_env,
                            is_mocked,
                            **c_exec_kwargs,
                        )
                        if bool(getattr(self._JSON_MODE_LOCAL, "exceeded_soft_timeout", False)):
                            compact_slow = True
                        c_out, c_cid = self._parse_cli_payload(c_proc)
                        compacted_ok = bool(c_out or c_cid)
                        if compacted_ok:
                            break
                        if enable_pre_quota and self._is_quota_or_credit_error(c_proc):
                            self._record_model_quota_exhausted(
                                compact_model,
                                model_unrecognized=self._is_unrecognized_model_error(c_proc),
                            )
                        if not is_mocked and compact_snap is not None:
                            restore_conversation_state(bot_cli_base, turn_cid, compact_snap)
                        if self._is_cli_interrupted(c_proc):
                            break
                    except subprocess.TimeoutExpired:
                        compact_slow = True
                        if not is_mocked and compact_snap is not None:
                            restore_conversation_state(bot_cli_base, turn_cid, compact_snap)
                        logger.warning(
                            "In-place /compact for topic '%s' conversation %s with model %s timed out after extended wait (soft=%ss); scheduling thread rotation for next turn.",
                            topic_key,
                            turn_cid,
                            compact_model,
                            compact_timeout,
                        )
                        break
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
            eff_schedule_rotation = bool(schedule_rotation or compact_slow)
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
                if eff_schedule_rotation:
                    record_kwargs["schedule_rotation"] = True
                elif count_slow_rotation:
                    record_kwargs["count_slow_rotation"] = True
                record_topic_turn(**record_kwargs)
                if main_turn_slow and topic_key:
                    slow_cnt = get_topic_slow_rotation_count(topic_key)
                    self._maybe_step_down_after_slow_rotations(
                        topic_key=topic_key,
                        slow_model=(
                            timed_out_model
                            or effective_model_used
                            or (models_to_try[attempt_idx] if attempt_idx < len(models_to_try) else "")
                        ),
                        models_to_try=models_to_try,
                        slow_count=slow_cnt,
                    )
            except Exception as e:
                logger.debug("Failed recording topic turn for %s: %s", topic_key, e)
        finally:
            self._JSON_MODE_LOCAL.include_repo_rules = prev_finalize_rules
            if release_lock and topic_lock is not None:
                topic_lock.release()

    def _query_cli_runner(
        self,
        prompt: str,
        *,
        selected_model: str,
        timeout_sec: float,
        cache_ttl_sec: float,
        now: float,
        thread_title: Optional[str],
        topic_key: Optional[str],
        conversation_id: Optional[str],
        resume_prompt: Optional[str],
        include_repo_rules: bool,
        require_json: bool,
        cli_mocked: bool,
    ) -> tuple[bool, Optional[str]]:
        """Invoke the authenticated local agy CLI runner; return (handled_terminal, response_text)."""
        topic_lock: Optional[threading.Lock] = None
        topic_lock_transferred = False
        prev_repo_rules = getattr(self._JSON_MODE_LOCAL, "include_repo_rules", False)
        self._JSON_MODE_LOCAL.include_repo_rules = include_repo_rules
        try:
            bot_cli_base = resolve_bot_cli_base()
            is_mocked = cli_mocked
            use_topic_session = bool(
                topic_key
                and (
                    not is_mocked
                    or hasattr(prepare_topic_session, "_mock_name")
                    or prepare_topic_session is not _ORIG_PREPARE_TOPIC_SESSION
                )
            )
            if use_topic_session and topic_key:
                t_key = str(topic_key).strip()
                with self._TOPIC_LOCKS_GUARD:
                    topic_lock = self._TOPIC_LOCKS.get(t_key)
                    if topic_lock is None:
                        topic_lock = threading.Lock()
                        self._TOPIC_LOCKS[t_key] = topic_lock
                topic_lock.acquire()
            session = TopicSession(conversation_id, "", False)
            if use_topic_session and not conversation_id:
                session = prepare_topic_session(topic_key, cli_base=bot_cli_base)

            effective_title = format_topic_turn_header(topic_key, thread_title) if topic_key else thread_title
            full_body = f"{session.history_prefix}{prompt}" if session.history_prefix else prompt
            full_cli_prompt = f"{effective_title}\n\n{full_body}" if effective_title else full_body
            use_warm_resume = bool(
                use_topic_session
                and topic_key
                and session.conversation_id
                and not session.history_prefix
                and isinstance(resume_prompt, str)
                and resume_prompt.strip()
                and is_topic_resume_warm(
                    topic_key,
                    cli_base=bot_cli_base,
                    conversation_id=session.conversation_id,
                )
            )
            cli_prompt = (
                (f"{effective_title}\n\n{resume_prompt}" if effective_title else resume_prompt)
                if use_warm_resume
                else full_cli_prompt
            )
            run_env = clone_process_env({"AGY_BOT_HOME": os.path.dirname(os.path.dirname(bot_cli_base))})

            raw_candidates = self.get_model_candidates(selected_model)
            pre_quota_fn = getattr(self.probe_live_model_quota, "__func__", self.probe_live_model_quota)
            pre_quota_mocked = (
                hasattr(self.probe_live_model_quota, "_mock_name")
                or "probe_live_model_quota" in self.__dict__
                or pre_quota_fn is not _ORIG_PROBE_LIVE_MODEL_QUOTA_FUNC
            )
            enable_pre_quota = not is_mocked or pre_quota_mocked
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
            detached_topic_prompt: Optional[str] = None

            def _resolve_detached_topic_prompt() -> str:
                nonlocal detached_topic_prompt
                if detached_topic_prompt is not None:
                    return detached_topic_prompt
                fp = full_cli_prompt
                if use_topic_session and topic_key and not session.history_prefix:
                    try:
                        hist_prefix = format_topic_history_prefix(topic_key)
                        if hist_prefix:
                            fb_body = f"{hist_prefix}{prompt}"
                            fp = f"{effective_title}\n\n{fb_body}" if effective_title else fb_body
                    except Exception:
                        pass
                detached_topic_prompt = fp
                return fp

            resume_abandoned = False
            had_resume_timeout = False
            timed_out_models: list[str] = []
            tried_detached_models: set[str] = set()
            exhausted_in_call: set[str] = set()
            # Derive abandoned-resume cap as 3/8 of primary budget (e.g. 45s @ 120s, 90s @ 240s; 5/8 + 3/8 = 1.0x)
            abandoned_resume_cap = timeout_sec * 0.375

            def _record_hard_timeout_slow_rotation() -> None:
                if not (
                    (had_resume_timeout or timed_out_models)
                    and use_topic_session
                    and topic_key
                    and (
                        not is_mocked
                        or hasattr(record_topic_resume_timeout, "_mock_name")
                        or record_topic_resume_timeout is not _ORIG_RECORD_TOPIC_RESUME_TIMEOUT
                        or hasattr(record_topic_slow_rotation, "_mock_name")
                        or record_topic_slow_rotation is not _ORIG_RECORD_TOPIC_SLOW_ROTATION
                    )
                ):
                    return
                try:
                    slow_cnt = record_topic_slow_rotation(
                        topic_key,
                        session.conversation_id,
                        cli_base=bot_cli_base,
                        rotate_immediately=True,
                    )
                    self._maybe_step_down_after_slow_rotations(
                        topic_key=topic_key,
                        slow_model=(
                            timed_out_models[0]
                            if timed_out_models
                            else (models_to_try[0] if models_to_try else selected_model)
                        ),
                        models_to_try=models_to_try,
                        slow_count=slow_cnt,
                    )
                except Exception as exc:
                    logger.debug("Failed recording topic slow rotation for %s: %s", topic_key, exc)

            for attempt_idx, model_candidate in enumerate(models_to_try):
                if resume_abandoned and model_candidate in tried_detached_models:
                    continue
                if attempt_idx > 0 and enable_pre_quota:
                    if model_candidate.strip().lower() in exhausted_in_call:
                        continue
                    rt_ex = (
                        self._QUOTA_MEM_CACHE.get("runtime_exhausted")
                        if isinstance(self._QUOTA_MEM_CACHE, dict)
                        and isinstance(self._QUOTA_MEM_CACHE.get("runtime_exhausted"), dict)
                        else {}
                    )
                    now_iter = time.time()
                    if safe_float(rt_ex.get(model_candidate), 0.0) > now_iter and any(
                        safe_float(rt_ex.get(rem_m), 0.0) <= now_iter
                        and rem_m.strip().lower() not in exhausted_in_call
                        for rem_m in models_to_try[attempt_idx + 1:]
                    ):
                        continue
                output, turn_cid = None, None
                effective_model_used = model_candidate
                replaced_conversation = False
                resume_timed_out = False
                exceeded_soft_timeout = False
                proc = None
                eff_conv_id = None if resume_abandoned else session.conversation_id
                eff_prompt = _resolve_detached_topic_prompt() if resume_abandoned else cli_prompt
                eff_timeout = (
                    min(timeout_sec, abandoned_resume_cap)
                    if (resume_abandoned and timeout_sec > abandoned_resume_cap)
                    else timeout_sec
                )
                pre_attempt_cids = (
                    self._list_conversation_cids(bot_cli_base)
                    if (not eff_conv_id and not is_mocked)
                    else set()
                )
                if eff_conv_id is None:
                    tried_detached_models.add(model_candidate)
                try:
                    exec_kwargs: Dict[str, Any] = {}
                    if use_topic_session and not resume_abandoned and not is_mocked:
                        exec_kwargs["allow_soft_extend"] = True
                    proc = self._execute_cli(
                        eff_conv_id, eff_prompt, eff_timeout, model_candidate,
                        bot_cli_base, run_env, is_mocked,
                        **exec_kwargs,
                    )
                    exceeded_soft_timeout = bool(getattr(self._JSON_MODE_LOCAL, "exceeded_soft_timeout", False))
                    output, turn_cid = self._parse_cli_payload(proc)
                    if resume_abandoned and output and use_topic_session and turn_cid:
                        replaced_conversation = turn_cid != session.conversation_id
                except subprocess.TimeoutExpired:
                    exceeded_soft_timeout = False
                    timed_out_models.append(model_candidate)
                    self._rollback_cli_attempt(
                        bot_cli_base, eff_conv_id, conv_snapshot if eff_conv_id else None, pre_attempt_cids, is_mocked
                    )
                    if not (eff_conv_id and not conversation_id):
                        if attempt_idx == len(models_to_try) - 1:
                            raise
                        logger.warning(
                            "Model candidate %s timed out after %ss; trying fallback.",
                            model_candidate,
                            eff_timeout,
                        )
                        continue
                    resume_timed_out = True
                    had_resume_timeout = True
                    resume_abandoned = True
                    logger.warning(
                        "Resuming topic '%s' conversation %s timed out after %ss; falling back to one-off thread.",
                        topic_key,
                        session.conversation_id,
                        eff_timeout,
                    )
                except Exception as exc:
                    exceeded_soft_timeout = False
                    self._rollback_cli_attempt(
                        bot_cli_base, eff_conv_id, conv_snapshot if eff_conv_id else None, pre_attempt_cids, is_mocked
                    )
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
                    self._rollback_cli_attempt(
                        bot_cli_base, eff_conv_id, conv_snapshot if eff_conv_id else None, pre_attempt_cids, is_mocked
                    )
                    output, turn_cid = None, None
                    continue

                if output is None:
                    self._rollback_cli_attempt(
                        bot_cli_base, eff_conv_id, conv_snapshot if eff_conv_id else None, pre_attempt_cids, is_mocked
                    )
                    if self._is_cli_interrupted(proc):
                        logger.info(
                            "agy CLI with model '%s' was interrupted (code %s); aborting LLM query without fallback.",
                            model_candidate,
                            proc.returncode if proc is not None else "unknown",
                        )
                        return True, None

                if (
                    output is None
                    and eff_conv_id
                    and not conversation_id
                    and not self._is_quota_or_credit_error(proc)
                ):
                    resume_abandoned = True
                    output, turn_cid, effective_model_used, replaced_conversation = self._attempt_detached_one_off_thread(
                        topic_key=topic_key,
                        session=session,
                        model_candidate=model_candidate,
                        models_to_try=models_to_try,
                        attempt_idx=attempt_idx,
                        resume_timed_out=resume_timed_out,
                        timeout_sec=timeout_sec,
                        fallback_prompt=_resolve_detached_topic_prompt(),
                        bot_cli_base=bot_cli_base,
                        run_env=run_env,
                        is_mocked=is_mocked,
                        require_json=require_json,
                        use_topic_session=use_topic_session,
                        tried_detached_models=tried_detached_models,
                    )

                if output:
                    self.last_model_used = effective_model_used
                    if attempt_idx > 0 or effective_model_used != selected_model:
                        logger.warning(
                            "LLM query succeeded using fallback model '%s' (primary '%s' unavailable/failed).",
                            effective_model_used,
                            selected_model,
                        )
                    self._cache_put(
                        selected_model,
                        prompt,
                        now=now,
                        text=output,
                        model_used=effective_model_used,
                        cache_ttl_sec=cache_ttl_sec,
                    )
                    if use_topic_session:
                        sched_rot = bool(exceeded_soft_timeout and not replaced_conversation)
                        count_slow = bool(timed_out_models and not sched_rot)
                        finalize_kwargs = dict(
                            topic_key=topic_key,
                            session=session,
                            turn_cid=turn_cid,
                            output=output,
                            replaced_conversation=replaced_conversation,
                            timeout_sec=timeout_sec,
                            models_to_try=models_to_try,
                            attempt_idx=attempt_idx,
                            enable_pre_quota=enable_pre_quota,
                            bot_cli_base=bot_cli_base,
                            run_env=run_env,
                            is_mocked=is_mocked,
                            include_repo_rules=include_repo_rules,
                            topic_lock=topic_lock,
                            schedule_rotation=sched_rot,
                            count_slow_rotation=count_slow,
                            timed_out_model=(timed_out_models[0] if timed_out_models else ""),
                            effective_model_used=effective_model_used,
                        )
                        if session.compact and turn_cid and not is_mocked and not sched_rot and not count_slow:
                            topic_lock_transferred = True
                            threading.Thread(
                                target=self._finalize_topic_turn,
                                kwargs={**finalize_kwargs, "release_lock": True},
                                daemon=False,
                            ).start()
                        else:
                            self._finalize_topic_turn(**finalize_kwargs, release_lock=False)
                    return True, output

                if enable_pre_quota and self._is_quota_or_credit_error(proc):
                    is_unrec = self._is_unrecognized_model_error(proc)
                    self._record_model_quota_exhausted(
                        model_candidate,
                        model_unrecognized=is_unrec,
                    )
                    if not is_unrec:
                        cand_lc = model_candidate.strip().lower()
                        p_gem, f_gem, c_mod = self._resolve_model_pools(extra_models=(model_candidate.strip(),))
                        pool = (
                            p_gem
                            if cand_lc in p_gem
                            else (f_gem if cand_lc in f_gem else (c_mod if cand_lc in c_mod else (cand_lc,)))
                        )
                        exhausted_in_call.update(m.lower() for m in pool)
                logger.warning(
                    "agy CLI with model '%s' failed (code %s): %s",
                    model_candidate,
                    proc.returncode if proc is not None else "timeout",
                    (proc.stderr or "").strip()[:200] if proc is not None else "",
                )
            _record_hard_timeout_slow_rotation()
        except subprocess.TimeoutExpired:
            _record_hard_timeout_slow_rotation()
            logger.warning("Gemini query timed out after %s seconds", timeout_sec)
            return True, None
        except Exception as e:
            logger.warning("Failed to invoke agy CLI: %s", e)
        finally:
            self._JSON_MODE_LOCAL.include_repo_rules = prev_repo_rules
            if topic_lock is not None and not topic_lock_transferred:
                topic_lock.release()
        return False, None

    def _query_openai_compat_fallback(
        self,
        prompt: str,
        *,
        selected_model: str = "",
        explicit_model: str = "",
        timeout_sec: float,
        cache_ttl_sec: float,
        now: float,
        require_json: bool,
    ) -> Optional[str]:
        """Query an optional OpenAI-compatible Chat Completions endpoint for resilience or A/B benchmarking."""
        cfg = resolve_openai_compat_config(
            url=self._openai_compat_url,
            api_key=self._openai_compat_api_key,
            models=self._openai_compat_models,
        )
        if cfg is None:
            return None
        text, model_id = query_openai_compat_chat(
            prompt,
            cfg,
            explicit_model=explicit_model,
            timeout_sec=timeout_sec,
            require_json=require_json,
            post_fn=fetch_json_post,
        )
        if not text:
            return None
        self.last_model_used = model_id
        self._cache_put(
            selected_model or model_id or "",
            prompt,
            now=now,
            text=text,
            model_used=model_id,
            cache_ttl_sec=cache_ttl_sec,
        )
        return text

    def _query_rest_fallback(
        self,
        prompt: str,
        *,
        selected_model: str,
        timeout_sec: float,
        cache_ttl_sec: float,
        now: float,
        require_json: bool,
    ) -> Optional[str]:
        """Query Gemini directly over REST API when an API key is configured and CLI failed."""
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
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
                }
                resp_data = fetch_json_post(url, body, timeout=timeout_sec)
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
                self.last_model_used = r_model
                self._cache_put(
                    selected_model,
                    prompt,
                    now=now,
                    text=text,
                    model_used=r_model,
                    cache_ttl_sec=cache_ttl_sec,
                )
                return text
            except Exception as e:
                logger.warning("Gemini REST API attempt with model %s failed: %s", r_model, e)
        return None

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
        resume_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Query Gemini with a prompt and return the text response."""
        self.last_model_used = None
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        now = time.time()
        explicit_call_model = model.strip() if isinstance(model, str) and model.strip() else None
        topic_preferred_model: Optional[str] = None
        include_repo_rules = False
        if topic_key:
            try:
                spec_obj = topic_spec(topic_key)
                include_repo_rules = bool(getattr(spec_obj, "can_self_heal", False))
                if not explicit_call_model and not getattr(self, "_explicit_model", False):
                    topic_preferred_model = spec_obj.preferred_model
            except Exception:
                topic_preferred_model = None
        selected_model = explicit_call_model or topic_preferred_model or self.model
        timeout_sec = safe_pos_float(timeout_sec, 15.0)
        cache_ttl_sec = safe_pos_float(cache_ttl_sec, 0.0)
        require_json = bool(getattr(self._JSON_MODE_LOCAL, "require_json", False))

        if (
            cached_txt := self._cache_get(
                selected_model,
                prompt,
                now=now,
                cache_ttl_sec=cache_ttl_sec,
                require_json=require_json,
                allow_legacy_key=not bool(explicit_call_model),
            )
        ) is not None:
            return cached_txt

        # 1. Custom runner (used for unit tests)
        if self._custom_runner is not None:
            try:
                res = self._custom_runner(prompt, selected_model, timeout_sec)
                if res:
                    self.last_model_used = selected_model
                self._cache_put(
                    selected_model,
                    prompt,
                    now=now,
                    text=res,
                    model_used=selected_model,
                    cache_ttl_sec=cache_ttl_sec,
                )
                return res
            except Exception as e:
                logger.error("Custom runner failed: %s", e)
                return None

        if is_gemini_llm_disabled():
            logger.debug("Gemini LLM query suppressed in backtest/replay mode.")
            return None

        allow_unmocked = is_unmocked_llm_query_allowed(getattr(self, "_allow_unmocked_queries", None))
        cli_mocked = (
            hasattr(subprocess.run, "_mock_name")
            or subprocess.run is not _ORIG_SUBPROCESS_RUN
            or hasattr(self._execute_cli, "_mock_name")
            or "_execute_cli" in self.__dict__
            or getattr(type(self), "_execute_cli", None) is not GeminiClient._execute_cli
        )
        rest_mocked = (
            hasattr(urllib.request.urlopen, "_mock_name")
            or urllib.request.urlopen is not _ORIG_URLOPEN
            or hasattr(fetch_json_post, "_mock_name")
            or fetch_json_post is not _ORIG_FETCH_JSON_POST
            or hasattr(query_openai_compat_chat, "_mock_name")
        )
        if not allow_unmocked and not cli_mocked and not rest_mocked:
            logger.debug("%s", "Gemini LLM external query suppressed by runtime policy.")
            return None

        # 1b. Direct OpenAI-compatible REST routing when explicitly targeted (e.g. "openai_compat:glm-5.3")
        is_explicit_openai, explicit_openai_slug = strip_openai_compat_prefix(selected_model)
        if is_explicit_openai:
            if allow_unmocked or rest_mocked:
                return self._query_openai_compat_fallback(
                    prompt,
                    selected_model=selected_model,
                    explicit_model=explicit_openai_slug,
                    timeout_sec=timeout_sec,
                    cache_ttl_sec=cache_ttl_sec,
                    now=now,
                    require_json=require_json,
                )
            return None

        # 2. Local authenticated CLI runner (subscription mode)
        if self.cli_path and os.path.exists(self.cli_path) and (allow_unmocked or cli_mocked):
            handled, cli_out = self._query_cli_runner(
                prompt,
                selected_model=selected_model,
                timeout_sec=timeout_sec,
                cache_ttl_sec=cache_ttl_sec,
                now=now,
                thread_title=thread_title,
                topic_key=topic_key,
                conversation_id=conversation_id,
                resume_prompt=resume_prompt,
                include_repo_rules=include_repo_rules,
                require_json=require_json,
                cli_mocked=cli_mocked,
            )
            if handled:
                return cli_out

        # 3a. Optional OpenAI-compatible REST resilience fallback (configured via OPENAI_COMPAT_FALLBACK_*)
        if allow_unmocked or rest_mocked:
            if openai_out := self._query_openai_compat_fallback(
                prompt,
                selected_model=selected_model,
                timeout_sec=timeout_sec,
                cache_ttl_sec=cache_ttl_sec,
                now=now,
                require_json=require_json,
            ):
                return openai_out

        # 3b. Direct Google AI Studio REST API fallback if API key is provided
        if self.api_key:
            return self._query_rest_fallback(
                prompt,
                selected_model=selected_model,
                timeout_sec=timeout_sec,
                cache_ttl_sec=cache_ttl_sec,
                now=now,
                require_json=require_json,
            )

        return None

    @staticmethod
    def _remove_detached_thread(bot_cli_base: str, cid: str) -> None:
        """Delete a one-off fallback conversation from the isolated bot store so orphans never accumulate."""
        if not bot_cli_base or not isinstance(cid, str):
            return
        cid = cid.strip()
        if not _DETACHED_CID_RE.fullmatch(cid):
            return
        delete_threads({cid}, bot_cli_base)

    def _cleanup_orphan_threads(
        self,
        bot_cli_base: str,
        pre_cids: set[str],
        *,
        keep_cid: Optional[str] = None,
    ) -> None:
        """Delete any newly created detached conversation threads not in ``pre_cids``."""
        for orphan_cid in self._list_conversation_cids(bot_cli_base) - pre_cids:
            if orphan_cid != keep_cid:
                self._remove_detached_thread(bot_cli_base, orphan_cid)

    def _rollback_cli_attempt(
        self,
        bot_cli_base: str,
        conversation_id: Optional[str],
        conv_snapshot: Optional[Dict[str, Optional[bytes]]],
        pre_attempt_cids: set[str],
        is_mocked: bool,
    ) -> None:
        """Restore ``conv_snapshot`` or remove orphan threads after a failed CLI attempt."""
        if is_mocked:
            return
        if conversation_id and conv_snapshot is not None:
            restore_conversation_state(bot_cli_base, conversation_id, conv_snapshot)
        elif not conversation_id:
            self._cleanup_orphan_threads(bot_cli_base, pre_attempt_cids)

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
        resume_prompt: Optional[str] = None,
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
                resume_prompt=resume_prompt,
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

    def query_topic_json(
        self,
        topic_key: str,
        prompt: str,
        *,
        subtitle: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        cache_ttl_sec: float = 0.0,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        resume_prompt: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query a registered topic (`TOPICS[topic_key]`) and return a validated JSON dictionary."""
        return query_topic_json(
            self,
            topic_key,
            prompt,
            subtitle=subtitle,
            timeout_sec=timeout_sec,
            cache_ttl_sec=cache_ttl_sec,
            conversation_id=conversation_id,
            model=model,
            resume_prompt=resume_prompt,
        )


def query_topic_json(
    client: Any,
    topic_key: str,
    prompt: str,
    *,
    subtitle: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    cache_ttl_sec: float = 0.0,
    conversation_id: Optional[str] = None,
    model: Optional[str] = None,
    resume_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Shared topic-aware JSON query dispatcher resolving title and timeout from ``topic_spec(topic_key)``."""
    if client is None:
        return None
    spec = topic_spec(topic_key)
    base_title = str(getattr(spec, "title", "") or topic_key).strip()
    sub = str(subtitle or "").strip()
    thread_title = f"{base_title} \u2014 {sub}" if sub else base_title
    default_timeout = safe_pos_float(
        getattr(spec, "default_timeout_sec", None),
        resolve_llm_timeout_sec(),
    )
    eff_timeout = safe_pos_float(timeout_sec, default_timeout)
    call_kwargs: Dict[str, Any] = {
        "timeout_sec": eff_timeout,
        "thread_title": thread_title,
        "topic_key": topic_key,
    }
    if cache_ttl_sec > 0.0:
        call_kwargs["cache_ttl_sec"] = cache_ttl_sec
    if conversation_id is not None:
        call_kwargs["conversation_id"] = conversation_id
    if isinstance(model, str) and model.strip():
        call_kwargs["model"] = model.strip()
    if isinstance(resume_prompt, str) and resume_prompt.strip():
        call_kwargs["resume_prompt"] = resume_prompt
    resp = client.query_json(prompt, **call_kwargs)
    return resp if isinstance(resp, dict) and resp else None
