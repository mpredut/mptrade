"""LLM quota probing, credit accounting, and dynamic model routing for GeminiClient."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from botcore import (
    _parse_bool_flag,
    bool_env,
    fetch_json_post,
    first_epoch_sec,
    first_pos_float,
    float_env,
    is_cache_fresh,
    is_unmocked_llm_query_allowed,
    normalize_epoch_sec,
    optional_env,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import LLM_QUOTA_STATUS_FILENAME, cache_path
from core.llm_topics import resolve_configured_llm_models
from state_io import generation_file_lock, read_json_dict, save_json_state

_agy_paths = importlib.import_module("orchestratorOS.agy.paths")
resolve_bot_cli_base = _agy_paths.resolve_bot_cli_base

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


_EFFORT_SUFFIXES: tuple[str, ...] = ("-xhigh", "-high", "-medium", "-low")
_FLASH_EFFORT_SUFFIXES: tuple[str, ...] = ("-high", "-medium", "-low")


def model_family_base(model_name: str) -> str:
    """Strip standard effort suffixes (-xhigh/-high/-medium/-low) to obtain the canonical model family."""
    cleaned = model_name.strip().lower() if isinstance(model_name, str) else ""
    for suffix in _EFFORT_SUFFIXES:
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def is_gemini_flash_model(model_name: str) -> bool:
    """Return True for Gemini Flash family slugs (eligible on both paid and free-tier pools)."""
    low = model_name.strip().lower() if isinstance(model_name, str) else ""
    return low.startswith("gemini-") and "flash" in low


def is_gemini_paid_model(model_name: str) -> bool:
    """Return True for non-Flash paid Gemini family slugs (e.g. Neon, Carbon, Pro, future series)."""
    low = model_name.strip().lower() if isinstance(model_name, str) else ""
    return low.startswith("gemini-") and not is_gemini_flash_model(low)


def is_claude_model(model_name: str) -> bool:
    """Return True for Anthropic Claude model slugs."""
    low = model_name.strip().lower() if isinstance(model_name, str) else ""
    return low.startswith("claude-")


def is_degraded_trading_model(model_name: Any) -> bool:
    """Return True for low-effort, Flash, or legacy REST models unsuitable for live enforce trading."""
    if not isinstance(model_name, str) or not model_name.strip():
        return False
    low = model_name.strip().lower()
    if low.startswith("openai_compat:"):
        low = low[len("openai_compat:") :].strip()
    return (
        low.endswith("-low")
        or is_gemini_flash_model(low)
        or low.startswith(("gemini-2.5", "gemini-2.0", "gemini-1.5"))
    )


def _expand_family_variants(
    models: Iterable[str],
    suffixes: tuple[str, ...] = _EFFORT_SUFFIXES,
) -> tuple[str, ...]:
    """Expand configured model slugs into their full family effort-tier set, preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    bases: list[str] = []
    for raw in models:
        m = raw.strip().lower() if isinstance(raw, str) else ""
        if not m:
            continue
        if m not in seen:
            seen.add(m)
            out.append(m)
        base = model_family_base(m)
        if base and base not in bases:
            bases.append(base)
    for base in bases:
        for suf in suffixes:
            cand = f"{base}{suf}"
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return tuple(out)


def resolve_quota_low_warn_fraction(default: float = 0.10) -> float:
    """Return clamped ``LLM_QUOTA_LOW_WARN_FRACTION`` in ``[0.0, 0.5]``."""
    raw = float_env("LLM_QUOTA_LOW_WARN_FRACTION", default=default)
    val = safe_float(raw, default)
    return max(0.0, min(0.5, val if val is not None else default))


def summarize_quota_pool_remaining(
    snapshot: Optional[Mapping[str, Any]],
    *,
    paid_gemini_models: Iterable[str],
    flash_gemini_models: Iterable[str],
    claude_models: Iterable[str],
) -> Dict[str, float]:
    """Compute remaining quota fractions for paid Gemini, Claude, and Flash pools plus top-up credits."""
    if not isinstance(snapshot, Mapping):
        return {"available_credits": 0.0, "primary_rem": 0.0, "claude_rem": 0.0, "flash_rem": 0.0}
    buckets = snapshot.get("buckets")
    b_map = buckets if isinstance(buckets, Mapping) else {}
    available_credits = safe_float(snapshot.get("available_credits"), 0.0)
    primary_rem = max((safe_float(b_map.get(m), 0.0) for m in paid_gemini_models), default=0.0)
    claude_keys = set(claude_models) | {k for k in b_map if is_claude_model(k)}
    claude_rem = max((safe_float(b_map.get(m), 0.0) for m in claude_keys), default=0.0)
    flash_keys = set(flash_gemini_models) | {k for k in b_map if is_gemini_flash_model(k)}
    flash_rem = max((safe_float(b_map.get(m), 0.0) for m in flash_keys), default=0.0)
    return {
        "available_credits": available_credits,
        "primary_rem": primary_rem,
        "claude_rem": claude_rem,
        "flash_rem": flash_rem,
    }


def evaluate_model_quota_escalation_allowed(
    quota_snapshot: Optional[Mapping[str, Any]],
    model_name: str,
    *,
    min_remaining_fraction: Optional[float] = None,
) -> tuple[bool, Optional[float], float]:
    """Return ``(allowed, remaining_fraction, threshold)`` for high-effort escalation to ``model_name``."""
    eff_thresh = (
        resolve_quota_low_warn_fraction()
        if min_remaining_fraction is None
        else max(0.0, min(1.0, safe_float(min_remaining_fraction, 0.10)))
    )
    cleaned_model = model_name.strip() if isinstance(model_name, str) else ""
    if not isinstance(quota_snapshot, Mapping) or not cleaned_model:
        return True, None, eff_thresh
    q_buckets = quota_snapshot.get("buckets") if isinstance(quota_snapshot.get("buckets"), Mapping) else {}
    q_cred = safe_float(quota_snapshot.get("available_credits"), default=0.0) or 0.0
    raw_rem = (
        q_buckets.get(cleaned_model)
        if cleaned_model in q_buckets
        else (q_buckets.get(cleaned_model.lower()) if cleaned_model.lower() in q_buckets else None)
    )
    rem = safe_float(raw_rem, default=None)
    if q_cred <= 0.0 and rem is not None and rem <= eff_thresh:
        return False, rem, eff_thresh
    return True, rem, eff_thresh


def canonical_rest_model(model_name: str) -> str:
    """Map arbitrary LLM model names (Gemini, Claude, etc.) to a canonical Google AI Studio REST model."""
    if not isinstance(model_name, str) or not model_name.strip():
        return "gemini-2.5-flash"
    base = model_family_base(model_name)
    if base.startswith(("gemini-2.5", "gemini-2.0", "gemini-1.5")):
        return base
    return "gemini-2.5-flash"


class GeminiQuotaRouterMixin:
    """Mixin providing Cloud Code quota/credit probing, alert dispatch, and config-driven model routing."""

    _QUOTA_LOCK = threading.Lock()
    _QUOTA_MEM_CACHE: Dict[str, Any] = {}

    model: str
    fallback_models: tuple[str, ...]
    _custom_fallbacks: bool

    @classmethod
    def _resolve_model_pools(
        cls,
        extra_models: Iterable[str] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Derive (paid_gemini_pool, flash_gemini_pool, claude_pool) dynamically from config.env."""
        configured = list(resolve_configured_llm_models())
        for raw in extra_models:
            if isinstance(raw, str) and raw.strip() and raw.strip() not in configured:
                configured.append(raw.strip())
        paid_gemini = _expand_family_variants(
            (m for m in configured if is_gemini_paid_model(m)),
            _EFFORT_SUFFIXES,
        )
        flash_gemini = _expand_family_variants(
            (m for m in configured if is_gemini_flash_model(m)),
            _FLASH_EFFORT_SUFFIXES,
        )
        claude_models = tuple(m.strip().lower() for m in configured if is_claude_model(m))
        return paid_gemini, flash_gemini, claude_models

    def get_model_candidates(self, primary_model: Optional[str] = None) -> list[str]:
        """Return an ordered, deduplicated sequence of models to attempt (primary -> fallbacks)."""
        selected = (primary_model.strip() if isinstance(primary_model, str) and primary_model.strip() else None) or self.model
        candidates = [selected]
        custom_fb = _parse_bool_flag(getattr(self, "_custom_fallbacks", False), False)
        if selected.endswith("-low") and not custom_fb:
            for fb in self.fallback_models:
                if fb and fb not in candidates and fb.endswith("-low"):
                    candidates.append(fb)
        if selected.endswith("-xhigh") and not custom_fb:
            high_step = f"{selected[:-6]}-high"
            if high_step not in candidates:
                candidates.append(high_step)
        for fb in self.fallback_models:
            if not fb or fb in candidates:
                continue
            if (
                selected.endswith(("-high", "-xhigh"))
                and not custom_fb
                and is_gemini_flash_model(fb)
                and fb.endswith("-medium")
            ):
                flash_high = f"{fb[:-7]}-high"
                if flash_high not in candidates:
                    candidates.append(flash_high)
            candidates.append(fb)
        return candidates

    def get_compact_model_candidates(
        self,
        models_to_try: list[str],
        attempt_idx: int = 0,
        quota_snapshot: Optional[Dict[str, Any]] = None,
    ) -> list[str]:
        """Return ordered models for in-place /compact summarization (configured LLM_COMPACT_MODEL first)."""
        idx = safe_nonneg_int(attempt_idx, default=0)
        failed_earlier = {m for m in models_to_try[:idx] if m}
        remaining = [m for m in models_to_try[idx:] if m]
        if _parse_bool_flag(getattr(self, "_custom_fallbacks", False), False):
            return remaining
        preferred_compact = (optional_env("LLM_COMPACT_MODEL", "") or "").strip()
        allowed = set(remaining)
        low_remaining = [m for m in remaining if m.endswith("-low")]
        ordered_lows = [
            *(m for m in low_remaining if is_gemini_flash_model(m)),
            *(m for m in low_remaining if is_gemini_paid_model(m)),
            *(m for m in low_remaining if not is_gemini_flash_model(m) and not is_gemini_paid_model(m)),
        ]
        compact_chain: list[str] = []
        if preferred_compact and preferred_compact not in failed_earlier:
            compact_chain.append(preferred_compact)
        for low_m in ordered_lows:
            if low_m and low_m not in compact_chain and (not allowed or low_m in allowed):
                compact_chain.append(low_m)
        for rem_m in remaining:
            if rem_m and rem_m not in compact_chain:
                compact_chain.append(rem_m)
        snap = quota_snapshot
        if snap is None and idx > 0 and isinstance(getattr(self, "_QUOTA_MEM_CACHE", None), dict):
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
        return _gc_attr("cache_path", cache_path)(LLM_QUOTA_STATUS_FILENAME)

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
        _quota_api_url = _gc_attr("QUOTA_API_URL", QUOTA_API_URL)
        _load_assist_api_url = _gc_attr("LOAD_ASSIST_API_URL", LOAD_ASSIST_API_URL)

        # Read strictly from the bot's own isolated CLI store so a separate human Web UI
        # account can never bleed its quota/credits into autonomous bot routing.
        target_cli_base = (
            bot_cli_base.strip()
            if isinstance(bot_cli_base, str) and bot_cli_base.strip()
            else _resolve_bot_cli_base()
        )
        access_token = ""
        if target_cli_base:
            tok_path = os.path.join(target_cli_base, "antigravity-oauth-token")
            tok_data = _read_json_dict(tok_path, default={})
            if tok_data:
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

        token_fp = hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16] if access_token else ""

        with cls._QUOTA_LOCK:
            for is_disk in (False, True):
                cand = _read_json_dict(state_path, default={}) if is_disk else cls._QUOTA_MEM_CACHE
                if (
                    isinstance(cand, dict)
                    and isinstance(cand.get("buckets"), dict)
                    and not (token_fp and (c_fp := str(cand.get("oauth_token_fp") or "").strip()) and token_fp != c_fp)
                    and is_cache_fresh(cand.get("checked_ts"), now, ttl_sec, force_refresh=force_refresh)
                ):
                    if is_disk:
                        cls._QUOTA_MEM_CACHE = dict(cand)
                    return dict(cand)


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
            # Cloud Code retrieveUserQuota may report paid Gemini and Flash Gemini buckets separately
            # or as a single shared Flash/Pro pool. Compute paid and Flash pools independently so a
            # 100% Flash bucket never masks an exhausted paid Gemini bucket, while preserving shared
            # pool inheritance when only legacy/tiered Flash buckets are advertised on a paid tier.
            gemini_pool_keys = [
                k for k in buckets if k.lower().startswith("gemini-") and not k.lower().endswith("-image")
            ]
            gemini_pool_rem = max(
                (safe_float(buckets.get(k), 0.0) for k in gemini_pool_keys),
                default=0.0,
            )
            paid_gemini_keys = [k for k in gemini_pool_keys if is_gemini_paid_model(k)]
            flash_gemini_keys = [k for k in gemini_pool_keys if is_gemini_flash_model(k)]
            flash_pool_rem = (
                max((safe_float(buckets.get(k), 0.0) for k in flash_gemini_keys), default=0.0)
                if flash_gemini_keys
                else gemini_pool_rem
            )
            paid_pool_rem = (
                max((safe_float(buckets.get(k), 0.0) for k in paid_gemini_keys), default=0.0)
                if paid_gemini_keys
                else gemini_pool_rem
            )
            is_free_tier = tier_id.lower() == "free-tier" and available_credits <= 0.0
            if gemini_pool_keys:
                paid_gemini_models, flash_gemini_models, _ = cls._resolve_model_pools()
                for flash_alias in flash_gemini_models:
                    buckets.setdefault(flash_alias, flash_pool_rem)
                for paid_alias in paid_gemini_models:
                    if is_free_tier:
                        paid_rem_val = 0.0
                    else:
                        alias_fam = model_family_base(paid_alias)
                        fam_keys = [k for k in paid_gemini_keys if model_family_base(k) == alias_fam]
                        paid_rem_val = (
                            max((safe_float(buckets.get(k), 0.0) for k in fam_keys), default=0.0)
                            if fam_keys
                            else paid_pool_rem
                        )
                    buckets.setdefault(paid_alias, paid_rem_val)
        except Exception as exc:
            logger.debug("Live LLM quota check failed (non-fatal): %s", exc)
            return None

        with cls._quota_state_rmw_lock(state_path, _save_json_state):
            prev = _read_json_dict(state_path, default={})
            if not isinstance(prev, dict):
                prev = {}
            prev_fp = str(prev.get("oauth_token_fp") or "").strip()
            same_oauth_account = not (token_fp and prev_fp and token_fp != prev_fp)
            runtime_exhausted: Dict[str, float] = {}
            prev_exhausted = prev.get("runtime_exhausted")
            if same_oauth_account and isinstance(prev_exhausted, dict) and available_credits <= 0.0:
                for m_name, exp_val in prev_exhausted.items():
                    exp_ts = safe_float(exp_val, 0.0)
                    if exp_ts > now:
                        runtime_exhausted[str(m_name)] = exp_ts

            snapshot: Dict[str, Any] = {
                "checked_ts": now,
                "oauth_token_fp": token_fp,
                "tier_id": tier_id,
                "available_credits": available_credits,
                "buckets": buckets,
                "runtime_exhausted": runtime_exhausted,
                "last_alert_ts": safe_float(prev.get("last_alert_ts"), 0.0) if same_oauth_account else 0.0,
                "last_alert_tier": str(prev.get("last_alert_tier") or "ok") if same_oauth_account else "ok",
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
        if isinstance(model_name, bool) or not isinstance(model_name, str):
            return
        cleaned = model_name.strip()
        if not cleaned:
            return
        unrec = _parse_bool_flag(model_unrecognized, False)
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
            paid_gemini_models, flash_gemini_models, _ = cls._resolve_model_pools(extra_models=(cleaned,))
            cleaned_lc = cleaned.lower()
            if unrec:
                affected: tuple[str, ...] = (cleaned,)
            elif cleaned_lc in paid_gemini_models:
                affected = paid_gemini_models
            elif cleaned_lc in flash_gemini_models:
                affected = flash_gemini_models
            else:
                affected = (cleaned,)
            for m_alias in affected:
                buckets[m_alias] = 0.0
                runtime_exhausted[m_alias] = exp_until
            data["buckets"] = buckets
            data["runtime_exhausted"] = runtime_exhausted
            data.setdefault("checked_ts", now)
            if unrec:
                data.setdefault("available_credits", 0.0)
            else:
                data["available_credits"] = 0.0
            cls._cache_and_save_quota_state_locked(state_path, data, _save_json_state)
            snapshot = dict(data)
        if not unrec and snapshot is not None:
            cls._maybe_notify_low_quota(snapshot)

    @classmethod
    def _maybe_notify_low_quota(cls, snapshot: Dict[str, Any]) -> None:
        """Send an alert on NTFY_TOPIC_ERROR when primary paid model quota/credits are near zero."""
        if not isinstance(snapshot, dict):
            return
        buckets = snapshot.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            return
        warn_threshold = resolve_quota_low_warn_fraction(default=0.10)
        cooldown_sec = max(
            300.0, safe_float(float_env("LLM_QUOTA_ALERT_COOLDOWN_SEC", default=14400.0), 14400.0)
        )

        paid_gemini_models, flash_gemini_models, claude_models = cls._resolve_model_pools()
        pool_rem = summarize_quota_pool_remaining(
            snapshot,
            paid_gemini_models=paid_gemini_models,
            flash_gemini_models=flash_gemini_models,
            claude_models=claude_models,
        )
        available_credits = pool_rem["available_credits"]
        primary_rem = pool_rem["primary_rem"]
        claude_rem = pool_rem["claude_rem"]
        flash_rem = pool_rem["flash_rem"]

        alert_updates: Dict[str, Any] = {}
        if available_credits > 0.0 or primary_rem > warn_threshold:
            if snapshot.get("last_alert_tier") not in (None, "", "ok"):
                alert_updates = {"last_alert_tier": "ok"}
        else:
            if claude_rem <= warn_threshold and flash_rem <= warn_threshold:
                alert_tier, severity_rank = "all_exhausted", 3
            elif claude_rem <= warn_threshold:
                alert_tier, severity_rank = "all_paid_low", 2
            else:
                alert_tier, severity_rank = "primary_low", 1

            rank_map = {
                "ok": 0,
                "primary_low": 1,
                "all_paid_low": 2,
                "all_exhausted": 3,
            }
            prev_tier = str(snapshot.get("last_alert_tier") or "ok")
            prev_rank = rank_map.get(prev_tier, 0)
            now = time.time()
            last_alert_ts = normalize_epoch_sec(snapshot.get("last_alert_ts"), default=0.0) or 0.0

            if severity_rank <= prev_rank and last_alert_ts > 0.0 and 0.0 <= (now - last_alert_ts) < cooldown_sec:
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

            primary_family = model_family_base(paid_gemini_models[0]) if paid_gemini_models else "gemini-primary"
            claude_family = model_family_base(claude_models[0]) if claude_models else "claude"
            flash_family = model_family_base(flash_gemini_models[0]) if flash_gemini_models else "gemini-flash"
            title = (
                "\U0001f6d1 ERROR: AI LLM Quota Near Zero"
                if primary_rem > 0.0
                else f"\U0001f6d1 ERROR: AI LLM Paid {primary_family} Quota Exhausted (0%)"
            )
            msg = (
                f"Available credits: {available_credits:.0f} | "
                f"{primary_family} (primary): {primary_rem * 100:.0f}% | "
                f"{claude_family} (paid fallback): {claude_rem * 100:.0f}% | "
                f"{flash_family} (fallback): {flash_rem * 100:.0f}%. "
                f"Automatic routing activated on first model with confirmed quota; "
                f"immediate dynamic recovery to {primary_family} upon top-up/reset."
            )
            try:
                watchdog_common.send_ntfy(title, msg)
                alert_updates = {"last_alert_ts": now, "last_alert_tier": alert_tier}
            except Exception as exc:
                logger.debug("Failed sending low-quota NTFY alert: %s", exc)

        if alert_updates:
            state_path = cls._get_quota_state_path()
            _read_json_dict, _save_json_state = (
                _gc_attr("read_json_dict", read_json_dict),
                _gc_attr("save_json_state", save_json_state),
            )
            with cls._quota_state_rmw_lock(state_path, _save_json_state):
                cur = _read_json_dict(state_path, default=None)
                merged = dict(cur) if isinstance(cur, dict) else dict(snapshot)
                merged.update(alert_updates)
                snapshot.update(alert_updates)
                cls._cache_and_save_quota_state_locked(state_path, merged, _save_json_state)
