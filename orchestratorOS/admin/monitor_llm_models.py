#!/usr/bin/env python3
"""4-hour LLM model, token, and credit health monitor.

Runs periodically (every 4 hours via cron: ``18 */4 * * *``):
  1. Probes live OAuth token / credits / per-model quota buckets via ``GeminiClient.probe_live_model_quota(force_refresh=True)``.
  2. Audits available CLI models against ``config.env`` via ``audit_llm_model_catalog_and_binaries(force_refresh=True)``.
  3. Automatically restores any latency-demoted primary models once tokens, credits, and model availability are confirmed healthy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import add_store_true_cli_args, load_dotenv, pos_float_env
from intelligence.sentiment.gemini_client import GeminiClient
from orchestratorOS.lib.technical_repair_ssot import audit_llm_model_catalog_and_binaries


def run_llm_model_monitor(
    workspace_dir: str = _ROOT,
    *,
    force_refresh: bool = True,
    notify: bool = True,
) -> dict[str, Any]:
    """Run one 4-hour LLM quota/token/model check and restore healthy latency-demoted models."""
    load_dotenv()
    monitor_interval_sec = pos_float_env("LLM_MODEL_MONITOR_INTERVAL_SEC", 14400.0)
    quota_snap = GeminiClient.probe_live_model_quota(
        force_refresh=force_refresh,
        restore_expired_demotions=False,
    )
    catalog_audit = audit_llm_model_catalog_and_binaries(
        workspace_dir,
        min_interval_sec=monitor_interval_sec,
        force_refresh=force_refresh,
        persist=True,
    )
    avail_models = (
        catalog_audit.get("available_models")
        if isinstance(catalog_audit, dict) and catalog_audit.get("agy_models_ok")
        else None
    )
    restored = GeminiClient.maybe_restore_latency_demoted_models(
        quota_snap,
        available_models=avail_models,
        force=True,
        notify=notify,
    )
    return {
        "checked_ts": time.time(),
        "quota_ok": isinstance(quota_snap, dict),
        "available_credits": (quota_snap or {}).get("available_credits") if isinstance(quota_snap, dict) else None,
        "available_models_count": len(avail_models) if isinstance(avail_models, list) else 0,
        "missing_configured_models": (
            list(catalog_audit.get("missing_configured_models") or [])
            if isinstance(catalog_audit, dict)
            else []
        ),
        "restored_models": restored,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="4-hour LLM model catalog, quota & primary-model recovery monitor.")
    add_store_true_cli_args(
        parser,
        ("--no-notify", "Suppress NTFY/email notifications"),
    )
    args = parser.parse_args(argv)
    summary = run_llm_model_monitor(_ROOT, force_refresh=True, notify=not args.no_notify)
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{ts_str}] [llm_model_monitor] quota_ok={summary['quota_ok']} "
        f"credits={summary['available_credits']} avail_models={summary['available_models_count']} "
        f"missing={summary['missing_configured_models']} restored={summary['restored_models']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
