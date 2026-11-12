"""LLM quota probing, credit accounting, and dynamic model routing for GeminiClient."""
from __future__ import annotations

import contextlib
import importlib
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from botcore import (
    bool_env,
    fetch_json_post,
    first_epoch_sec,
    first_pos_float,
    float_env,
    is_cache_fresh,
    is_unmocked_llm_query_allowed,
    optional_env,
    resolve_patched_attr,
    safe_float,
    safe_pos_float,
)
from cachepaths import cache_path
from state_io import generation_file_lock, read_json_dict, save_json_state

_agy_paths = importlib.import_module("orchestratorOS.agy.paths")
resolve_bot_cli_base = _agy_paths.resolve_bot_cli_base
resolve_ui_cli_base = _agy_paths.resolve_ui_cli_base

logger = logging.getLogger("intelligence.sentiment.gemini_client")

QUOTA_API_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
LOAD_ASSIST_API_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
AGY_QUOTA_USER_AGENT = "antigravity/1.4.12"

_ORIG_GC_DEFAULTS: Dict[str, Any] = {
    "cache_path": cache_path,
    "read_json_dict": read_json_dict,
    "save_json_state": save_json_state,
    "fetch_json_post": fetch_json_post,
    "resolve_bot_cli_base": resolve_bot_cli_base,
    "resolve_ui_cli_base": resolve_ui_cli_base,
    "QUOTA_API_URL": QUOTA_API_URL,
    "LOAD_ASSIST_API_URL": LOAD_ASSIST_API_URL,
}


def _gc_attr(attr_name: str, current_val: Any) -> Any:
    orig = _ORIG_GC_DEFAULTS.get(attr_name, current_val)
    return resolve_patched_attr(
        attr_name,
        orig,
        local_val=current_val,
        defaults=(orig,),
        modules=("intelligence.sentiment.gemini_client", __name__),
    )


def canonical_rest_model(model_name: str) -> str:
    """Map arbitrary LLM model names (Gemini, Argon, Claude) to a canonical Google AI Studio REST model."""
    if not isinstance(model_name, str) or not model_name.strip():
        return "gemini-2.5-flash"
    cleaned = model_name.strip().lower()
    base = cleaned.replace("-low", "").replace("-medium", "").replace("-high", "")
    if base.startswith(("gemini-2.5", "gemini-2.0", "gemini-1.5")):
        return base
    return "gemini-2.5-flash"


class GeminiQuotaRouterMixin:
    """Mixin providing Cloud Code quota/credit probing, alert dispatch, and model candidate routing."""

    _QUOTA_LOCK = threading.Lock()
    _QUOTA_MEM_CACHE: Dict[str, Any] = {}
    _ARGON_POOL_MODELS = (
        "gemini-4-argon-high",
        "gemini-4-argon-medium",
        "gemini-4-argon-low",
    )
    _FLASH_POOL_MODELS = (
        "gemini-3.8-flash-high",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-low",
    )
    _SHARED_GEMINI_POOL_MODELS = _ARGON_POOL_MODELS + _FLASH_POOL_MODELS

    model: str
    fallback_models: tuple[str, ...]
    _custom_fallbacks: bool

    def get_model_candidates(self, primary_model: Optional[str] = None) -> list[str]:
        """Return an ordered, deduplicated sequence of models to attempt (primary -> fallbacks)."""
        selected = (primary_model.strip() if isinstance(primary_model, str) and primary_model.strip() else None) or self.model
        candidates = [selected]
        custom_fb = getattr(self, "_custom_fallbacks", False)
        if selected.endswith("-low") and not custom_fb:
            for fb in self.fallback_models:
                if fb and fb not in candidates and fb.endswith("-low"):
                    candidates.append(fb)
        for fb in self.fallback_models:
            if not fb or fb in candidates:
                continue
            if selected.endswith("-high") and not custom_fb and fb == "gemini-3.8-flash-medium":
                if "gemini-3.8-flash-high" not in candidates:
                    candidates.append("gemini-3.8-flash-high")
            candidates.append(fb)
        return candidates

    def get_compact_model_candidates(
        self,
        models_to_try: list[str],
        attempt_idx: int = 0,
        quota_snapshot: Optional[Dict[str, Any]] = None,
    ) -> list[str]:
        """Return ordered models for in-place /compact summarization (fast low-cost Gemini Flash model first)."""
        remaining = [m for m in models_to_try[attempt_idx:] if m]
        if getattr(self, "_custom_fallbacks", False):
            return remaining
        preferred_compact = optional_env("LLM_COMPACT_MODEL", "gemini-3.8-flash-low")
        allowed = set(remaining)
        compact_chain: list[str] = []
        for low_m in (
            preferred_compact,
            "gemini-3.8-flash-low",
            "gemini-4-argon-low",
            "claude-sonnet-5-5-low",
        ):
            if low_m and low_m not in compact_chain and (not allowed or low_m in allowed):
                compact_chain.append(low_m)
        for rem_m in remaining:
            if rem_m and rem_m not in compact_chain:
                compact_chain.append(rem_m)
        snap = quota_snapshot
        if snap is None and attempt_idx > 0 and isinstance(getattr(self, "_QUOTA_MEM_CACHE", None), dict):
            snap = self._QUOTA_MEM_CACHE
        if isinstance(snap, dict) and compact_chain:
            now = time.time()
            avail_cred = safe_float(snap.get("available_credits"), 0.0)
            rt_ex = snap.get("runtime_exhausted") if isinstance(snap.get("runtime_exhausted"), dict) else None
            if isinstance(rt_ex, dict) and rt_ex and avail_cred <= 0.0:
                non_exhausted = [m for m in compact_chain if safe_float(rt_ex.get(m), 0.0) <= now]
                if non_exhausted:
                    compact_chain = non_exhausted
        return compact_chain or remaining

    @staticmethod
    def _get_quota_state_path() -> str:
        """Return shared SSOT cache path for live LLM model quota & credit status."""
        return _gc_attr("cache_path", cache_path)("llm_quota_status.json")

    @staticmethod
    def _extract_available_credits(load_assist_payload: Any) -> float:
        """Extract total available paid credits from loadCodeAssist response payload."""
        if not isinstance(load_assist_payload, dict):
            return 0.0
        total = 0.0
        seen_tier_ids: set[str] = set()
        for tier_key in ("paidTier", "currentTier"):
            tier = load_assist_payload.get(tier_key)
            if not isinstance(tier, dict):
                continue
            tid = str(tier.get("id") or "").strip()
            if tid and tid in seen_tier_ids:
                continue
            if not tid and total > 0.0:
                continue
            credits_list = tier.get("availableCredits")
            if not isinstance(credits_list, list):
                continue
            if tid:
                seen_tier_ids.add(tid)
            for item in credits_list:
                if isinstance(item, dict):
                    rem_f = (
                        safe_float(item.get("remainingCredits"), default=None)
                        if item.get("remainingCredits") is not None
                        else None
                    )
                    val = (
                        max(0.0, rem_f)
                        if rem_f is not None
                        else (
                            first_pos_float(
                                item.get("creditAmount"),
                                item.get("amount"),
                                item.get("credits"),
                                default=0.0,
                            )
                            or 0.0
                        )
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
        ttl_sec = max(5.0, float_env("LLM_QUOTA_CACHE_TTL_SEC", default=30.0) or 30.0)
        state_path = cls._get_quota_state_path()
        _read_json_dict = _gc_attr("read_json_dict", read_json_dict)
        _save_json_state = _gc_attr("save_json_state", save_json_state)
        _fetch_json_post = _gc_attr("fetch_json_post", fetch_json_post)
        _resolve_bot_cli_base = _gc_attr("resolve_bot_cli_base", resolve_bot_cli_base)
        _resolve_ui_cli_base = _gc_attr("resolve_ui_cli_base", resolve_ui_cli_base)
        _quota_api_url = _gc_attr("QUOTA_API_URL", QUOTA_API_URL)
        _load_assist_api_url = _gc_attr("LOAD_ASSIST_API_URL", LOAD_ASSIST_API_URL)

        with cls._QUOTA_LOCK:
            mem = cls._QUOTA_MEM_CACHE
            if (
                isinstance(mem, dict)
                and isinstance(mem.get("buckets"), dict)
                and is_cache_fresh(mem.get("checked_ts"), now, ttl_sec, force_refresh=force_refresh)
            ):
                return dict(mem)

            disk = _read_json_dict(state_path, default={})
            if (
                isinstance(disk, dict)
                and isinstance(disk.get("buckets"), dict)
                and is_cache_fresh(disk.get("checked_ts"), now, ttl_sec, force_refresh=force_refresh)
            ):
                cls._QUOTA_MEM_CACHE = dict(disk)
                return dict(disk)

        cli_bases = []
        if bot_cli_base and isinstance(bot_cli_base, str):
            cli_bases.append(bot_cli_base)
        for resolved_base in (_resolve_bot_cli_base(), _resolve_ui_cli_base()):
            if resolved_base and resolved_base not in cli_bases:
                cli_bases.append(resolved_base)

        access_token = ""
        for base_dir in cli_bases:
            tok_path = os.path.join(base_dir, "antigravity-oauth-token")
            tok_data = _read_json_dict(tok_path, default={})
            if not tok_data:
                continue
            nested_tok = tok_data.get("token") if isinstance(tok_data.get("token"), dict) else {}
            tok = str(nested_tok.get("access_token") or tok_data.get("access_token") or "").strip()
            exp_ts = (
                first_epoch_sec(
                    nested_tok.get("expiry_timestamp"),
                    tok_data.get("expiry_timestamp"),
                    default=0.0,
                    allow_iso=True,
                )
                or 0.0
            )
            if tok and (exp_ts <= 0.0 or exp_ts > now + 10.0):
                access_token = tok
                break

        if not access_token:
            return None

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": AGY_QUOTA_USER_AGENT,
        }

        available_credits = 0.0
        tier_id = ""
        companion_project = ""
        try:
            assist_payload = _fetch_json_post(_load_assist_api_url, timeout=2.0, headers=headers)
            if isinstance(assist_payload, dict):
                available_credits = cls._extract_available_credits(assist_payload)
                paid_tier = assist_payload.get("paidTier") if isinstance(assist_payload.get("paidTier"), dict) else {}
                curr_tier = assist_payload.get("currentTier") if isinstance(assist_payload.get("currentTier"), dict) else {}
                tier_id = str(paid_tier.get("id") or curr_tier.get("id") or "").strip()
                companion_project = str(assist_payload.get("cloudaicompanionProject") or "").strip()
        except Exception as exc:
            logger.debug("Live LLM loadCodeAssist credit check failed (non-fatal): %s", exc)

        buckets: Dict[str, float] = {}
        try:
            if companion_project:
                quota_payload = _fetch_json_post(
                    _quota_api_url,
                    {"project": companion_project},
                    timeout=2.5,
                    headers=headers,
                )
            else:
                quota_payload = _fetch_json_post(_quota_api_url, timeout=2.5, headers=headers)
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
            # On free-tier without top-up credits, Argon slugs are unavailable in agy CLI,
            # so only Flash aliases inherit the free-tier Gemini pool fraction.
            gemini_pool_keys = ("gemini-3.8-flash-tiered", "gemini-2.5-pro", "gemini-3-flash")
            gemini_pool_rem = max(
                (safe_float(buckets.get(k), 0.0) for k in gemini_pool_keys),
                default=0.0,
            )
            is_free_tier = tier_id.lower() == "free-tier" and available_credits <= 0.0
            if gemini_pool_rem > 0.0 or any(k in buckets for k in gemini_pool_keys):
                for flash_alias in cls._FLASH_POOL_MODELS:
                    buckets.setdefault(flash_alias, gemini_pool_rem)
                argon_rem_val = 0.0 if is_free_tier else gemini_pool_rem
                for argon_alias in cls._ARGON_POOL_MODELS:
                    buckets.setdefault(argon_alias, argon_rem_val)
        except Exception as exc:
            logger.debug("Live LLM quota check failed (non-fatal): %s", exc)
            return None

        with cls._quota_state_rmw_lock(state_path, _save_json_state):
            prev = _read_json_dict(state_path, default={})
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
            cls._cache_and_save_quota_state_locked(state_path, snapshot, _save_json_state)

        cls._maybe_notify_low_quota(snapshot)
        return snapshot

    @classmethod
    @contextlib.contextmanager
    def _quota_state_rmw_lock(cls, state_path: str, save_fn: Any):
        with cls._QUOTA_LOCK, (
            generation_file_lock(state_path) if save_fn is save_json_state else contextlib.nullcontext()
        ):
            yield

    @classmethod
    def _cache_and_save_quota_state_locked(
        cls, state_path: str, snapshot: Dict[str, Any], save_fn: Any
    ) -> None:
        cls._QUOTA_MEM_CACHE = dict(snapshot)
        try:
            save_fn(state_path, snapshot)
        except Exception:
            pass

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
        if confirmed:
            return confirmed
        non_exhausted = [m for m in candidates if safe_float(runtime_exhausted.get(m), 0.0) <= now]
        return non_exhausted or list(candidates)

    @classmethod
    def _record_model_quota_exhausted(
        cls,
        model_name: str,
        ttl_sec: float = 120.0,
        *,
        model_unrecognized: bool = False,
    ) -> None:
        """Record a runtime quota/credit error or model unavailability and trigger low-quota alert check."""
        cleaned = str(model_name or "").strip()
        if not cleaned:
            return
        now = time.time()
        exp_until = now + max(15.0, safe_pos_float(ttl_sec, default=120.0))
        state_path, _read_json_dict, _save_json_state = (
            cls._get_quota_state_path(),
            _gc_attr("read_json_dict", read_json_dict),
            _gc_attr("save_json_state", save_json_state),
        )
        snapshot: Optional[Dict[str, Any]] = None
        with cls._quota_state_rmw_lock(state_path, _save_json_state):
            data = _read_json_dict(state_path, default={})
            if not isinstance(data, dict):
                data = {}
            buckets = data.get("buckets") if isinstance(data.get("buckets"), dict) else {}
            runtime_exhausted = (
                data.get("runtime_exhausted")
                if isinstance(data.get("runtime_exhausted"), dict)
                else {}
            )
            affected = (
                (cleaned,)
                if model_unrecognized
                else (
                    cls._SHARED_GEMINI_POOL_MODELS
                    if cleaned in cls._SHARED_GEMINI_POOL_MODELS
                    else (cleaned,)
                )
            )
            for m_alias in affected:
                buckets[m_alias] = 0.0
                runtime_exhausted[m_alias] = exp_until
            data["buckets"] = buckets
            data["runtime_exhausted"] = runtime_exhausted
            data.setdefault("checked_ts", now)
            if model_unrecognized:
                data.setdefault("available_credits", 0.0)
            else:
                data["available_credits"] = 0.0
            cls._cache_and_save_quota_state_locked(state_path, data, _save_json_state)
            snapshot = dict(data)
        if not model_unrecognized and snapshot is not None:
            cls._maybe_notify_low_quota(snapshot)

    @classmethod
    def _maybe_notify_low_quota(cls, snapshot: Dict[str, Any]) -> None:
        """Send an alert on NTFY_TOPIC_ERROR when primary paid model quota/credits are near zero."""
        if not isinstance(snapshot, dict):
            return
        buckets = snapshot.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            return
        _save_json_state = _gc_attr("save_json_state", save_json_state)
        available_credits = safe_float(snapshot.get("available_credits"), 0.0)
        warn_threshold = max(
            0.0, min(0.5, safe_float(float_env("LLM_QUOTA_LOW_WARN_FRACTION", default=0.10), 0.10))
        )
        cooldown_sec = max(
            300.0, safe_float(float_env("LLM_QUOTA_ALERT_COOLDOWN_SEC", default=14400.0), 14400.0)
        )

        argon_rem = max(
            safe_float(buckets.get("gemini-4-argon-medium"), 0.0),
            safe_float(buckets.get("gemini-4-argon-low"), 0.0),
            safe_float(buckets.get("gemini-4-argon-high"), 0.0),
        )
        claude_rem = max(
            safe_float(buckets.get("claude-sonnet-5-5-medium"), 0.0),
            safe_float(buckets.get("claude-sonnet-5-5-low"), 0.0),
        )
        flash_rem = max(
            safe_float(buckets.get("gemini-3.8-flash-high"), 0.0),
            safe_float(buckets.get("gemini-3.8-flash-medium"), 0.0),
            safe_float(buckets.get("gemini-3.8-flash-low"), 0.0),
            safe_float(buckets.get("gemini-3.8-flash-tiered"), 0.0),
        )

        if available_credits > 0.0 or argon_rem > warn_threshold:
            if snapshot.get("last_alert_tier") not in (None, "", "ok"):
                state_path = cls._get_quota_state_path()
                with cls._quota_state_rmw_lock(state_path, _save_json_state):
                    snapshot["last_alert_tier"] = "ok"
                    cls._cache_and_save_quota_state_locked(state_path, snapshot, _save_json_state)
            return

        if claude_rem <= warn_threshold and flash_rem <= warn_threshold:
            alert_tier, severity_rank = "all_exhausted", 3
        elif claude_rem <= warn_threshold:
            alert_tier, severity_rank = "all_paid_low", 2
        else:
            alert_tier, severity_rank = "argon_low", 1

        rank_map = {"ok": 0, "argon_low": 1, "all_paid_low": 2, "all_exhausted": 3}
        prev_tier = str(snapshot.get("last_alert_tier") or "ok")
        prev_rank = rank_map.get(prev_tier, 0)
        now = time.time()
        last_alert_ts = safe_float(snapshot.get("last_alert_ts"), 0.0)

        if severity_rank <= prev_rank and (now - last_alert_ts) < cooldown_sec:
            return

        try:
            watchdog_common = importlib.import_module("orchestratorOS.lib.watchdog_common")
        except Exception:
            return

        ntfy_mocked = (
            hasattr(watchdog_common.send_ntfy, "_mock_name")
            or getattr(watchdog_common.send_ntfy, "__module__", "") != "orchestratorOS.lib.watchdog_common"
        )
        if (
            not is_unmocked_llm_query_allowed()
            and not ntfy_mocked
            and not bool_env("LLM_FORCE_QUOTA_ALERT_TEST", default=False)
        ):
            return

        title = (
            "🛑 ERROR: AI LLM Quota Near Zero"
            if argon_rem > 0.0
            else "🛑 ERROR: AI LLM Paid Argon Quota Exhausted (0%)"
        )
        msg = (
            f"Available credits: {available_credits:.0f} | "
            f"gemini-4-argon (med/low/high): {argon_rem * 100:.0f}% | "
            f"claude-sonnet-5-5 (paid fallback): {claude_rem * 100:.0f}% | "
            f"gemini-3.8-flash (fallback): {flash_rem * 100:.0f}%. "
            f"Automatic routing activated on first model with confirmed quota; "
            f"immediate dynamic recovery to Argon upon top-up/reset."
        )
        dispatched = False
        try:
            watchdog_common.send_ntfy(title, msg)
            dispatched = True
        except Exception as exc:
            logger.debug("Failed sending low-quota NTFY alert: %s", exc)

        if dispatched:
            state_path = cls._get_quota_state_path()
            with cls._quota_state_rmw_lock(state_path, _save_json_state):
                snapshot["last_alert_ts"] = now
                snapshot["last_alert_tier"] = alert_tier
                cls._cache_and_save_quota_state_locked(state_path, snapshot, _save_json_state)
