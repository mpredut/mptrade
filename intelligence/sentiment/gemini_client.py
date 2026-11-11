"""Google Gemini client providing LLM reasoning via authenticated CLI or REST API."""
from __future__ import annotations

import fcntl
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import urllib.request

from botcore import (
    _parse_bool_flag,
    fetch_json_post,
    is_cache_fresh,
    is_unmocked_llm_query_allowed,
    load_dotenv,
    optional_env,
    required_env,
    safe_float,
    safe_pos_float,
    split_csv,
    topic_spec,
)
from cachepaths import cache_path
from orchestratorOS.agy.paths import resolve_agy_bot_bin, resolve_bot_cli_base, resolve_ui_cli_base
from orchestratorOS.agy.summary_store import (
    delete_threads,
    restore_conversation_state,
    snapshot_conversation_state,
)
from orchestratorOS.agy.topics import (
    TopicSession,
    condense_decision,
    format_topic_history_prefix,
    format_topic_turn_header,
    prepare_topic_session,
    record_topic_turn,
)
from intelligence.sentiment.llm_quota_router import (
    LOAD_ASSIST_API_URL,
    QUOTA_API_URL,
    GeminiQuotaRouterMixin,
    canonical_rest_model,
)
from state_io import extract_json_object, parse_json_payload, read_json_dict, save_json_state

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
        self._cache: Dict[str, tuple[float, str]] = {}

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
    ) -> subprocess.CompletedProcess:
        """Run the isolated agy CLI tool with robust file-locking and timeout management.

        Non-coding autonomous queries (trading, macro, sentiment, pre-trade guards) run with
        ``include_repo_rules=False`` from ``bot_cli_base`` so ``agy`` does not walk up into
        ``_ROOT_DIR`` or inject ``AGENTS.md`` engineering rules into financial reasoning prompts.
        Code self-healing topics (``can_self_heal=True``) set ``include_repo_rules=True`` to run
        inside ``_ROOT_DIR`` with ``--add-dir`` and load ``AGENTS.md``.
        """
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
            pre_dirty = (
                self._snapshot_dirty_tracked_files(_ROOT_DIR)
                if (use_repo_rules and os.path.isdir(_ROOT_DIR))
                else None
            )
            try:
                return subprocess.run(cmd, capture_output=True, text=True, timeout=call_timeout,
                                      check=False, env=run_env, cwd=exec_cwd)
            finally:
                if pre_dirty is not None:
                    self._restore_newly_dirtied_tracked_files(_ROOT_DIR, pre_dirty)
                if locked:
                    try:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass

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

    @classmethod
    def _is_quota_or_credit_error(cls, proc: Optional[subprocess.CompletedProcess]) -> bool:
        """Return True when the CLI failed due to exhausted AI credits, quota, rate limits, or unrecognized model."""
        if proc is None or cls._is_cli_interrupted(proc):
            return False
        err_raw = (proc.stderr or "").strip()
        out_raw = (proc.stdout or "").strip()
        # Inspect stderr when present; only fall back to stdout when stderr is empty
        # so buffered prompt/response text on stdout (e.g. "rate limit") is never mistaken for a CLI error.
        combined = (err_raw if err_raw else out_raw).lower()
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
        )
        if not allow_unmocked and not cli_mocked and not rest_mocked:
            logger.debug("Gemini LLM external query suppressed by runtime policy.")
            return None

        # 2. Local authenticated CLI runner (subscription mode)
        if self.cli_path and os.path.exists(self.cli_path) and (allow_unmocked or cli_mocked):
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

                # The per-call title is a turn header; the thread itself keeps its permanent topic name.
                effective_title = format_topic_turn_header(topic_key, thread_title) if topic_key else thread_title
                body = f"{session.history_prefix}{prompt}" if session.history_prefix else prompt
                cli_prompt = f"{effective_title}\n\n{body}" if effective_title else body

                run_env = os.environ.copy()
                run_env["AGY_BOT_HOME"] = os.path.dirname(os.path.dirname(bot_cli_base))

                # Attempt primary model followed by fallback models if CLI call fails
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

                for attempt_idx, model_candidate in enumerate(models_to_try):
                    if attempt_idx > 0 and enable_pre_quota:
                        rt_ex = (
                            self._QUOTA_MEM_CACHE.get("runtime_exhausted")
                            if isinstance(self._QUOTA_MEM_CACHE, dict)
                            and isinstance(self._QUOTA_MEM_CACHE.get("runtime_exhausted"), dict)
                            else {}
                        )
                        now_iter = time.time()
                        if safe_float(rt_ex.get(model_candidate), 0.0) > now_iter and any(
                            safe_float(rt_ex.get(rem_m), 0.0) <= now_iter
                            for rem_m in models_to_try[attempt_idx + 1:]
                        ):
                            continue
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
                        self._rollback_cli_attempt(
                            bot_cli_base, session.conversation_id, conv_snapshot, pre_attempt_cids, is_mocked
                        )
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
                        self._rollback_cli_attempt(
                            bot_cli_base, session.conversation_id, conv_snapshot, pre_attempt_cids, is_mocked
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
                            bot_cli_base, session.conversation_id, conv_snapshot, pre_attempt_cids, is_mocked
                        )
                        output = None
                        turn_cid = None
                        continue

                    if output is None:
                        self._rollback_cli_attempt(
                            bot_cli_base, session.conversation_id, conv_snapshot, pre_attempt_cids, is_mocked
                        )
                        if self._is_cli_interrupted(proc):
                            logger.info(
                                "agy CLI with model '%s' was interrupted (code %s); aborting LLM query without fallback.",
                                model_candidate,
                                proc.returncode if proc is not None else "unknown",
                            )
                            return None

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
                                self._cleanup_orphan_threads(
                                    bot_cli_base, pre_cids, keep_cid=session.conversation_id
                                )
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
                                prev_finalize_rules = getattr(self._JSON_MODE_LOCAL, "include_repo_rules", False)
                                self._JSON_MODE_LOCAL.include_repo_rules = include_repo_rules
                                try:
                                    compacted_ok = False
                                    if session.compact and turn_cid and not replaced_conversation:
                                        compact_timeout = max(timeout_sec, 120.0)
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
                                                c_proc = self._execute_cli(
                                                    turn_cid,
                                                    compact_prompt,
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
                                                if enable_pre_quota and self._is_quota_or_credit_error(c_proc):
                                                    self._record_model_quota_exhausted(compact_model)
                                                if not is_mocked and compact_snap is not None:
                                                    restore_conversation_state(bot_cli_base, turn_cid, compact_snap)
                                                if self._is_cli_interrupted(c_proc):
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
                                    self._JSON_MODE_LOCAL.include_repo_rules = prev_finalize_rules
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
                self._JSON_MODE_LOCAL.include_repo_rules = prev_repo_rules
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

    def query_topic_json(
        self,
        topic_key: str,
        prompt: str,
        *,
        subtitle: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        cache_ttl_sec: float = 0.0,
        conversation_id: Optional[str] = None,
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
) -> Optional[Dict[str, Any]]:
    """Shared topic-aware JSON query dispatcher resolving title and timeout from ``topic_spec(topic_key)``."""
    if client is None:
        return None
    spec = topic_spec(topic_key)
    base_title = str(getattr(spec, "title", "") or topic_key).strip()
    sub = str(subtitle or "").strip()
    thread_title = f"{base_title} — {sub}" if sub else base_title
    default_timeout = safe_pos_float(
        getattr(spec, "default_timeout_sec", None),
        120.0 if getattr(spec, "can_self_heal", False) else 90.0,
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
    resp = client.query_json(prompt, **call_kwargs)
    return resp if isinstance(resp, dict) and resp else None
