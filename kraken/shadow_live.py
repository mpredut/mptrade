#!/usr/bin/env python3
"""kraken/shadow_live.py — LIVE SHADOW TEST (read-only, ZERO real orders, bot-isolated).

Run configurations through the faithful engine: replay.run_replay makes the same
decisions as live Strategy.step over the SAME live OHLC downloaded from Kraken, and log
comparative paper P&L. This forward-tests the PRODUCTION configuration against shadow
candidates preregistered from research before risking real money:

  - current: exact LIVE configuration loaded from .env and then config.env; reference
  - pre0923: the live configuration before the 23 Sep 2026 promotion (TP 5%, no DCA
    spacing growth, fixed 3% trail, no regime gate, no profit floor, 12.5% stop); the
    control that tells whether that promotion helps going forward
  - rev_*: live with exactly ONE of the six 23 Sep changes reverted (tp5, spacing0,
    trail_fixed, gate_off, floor0, sl125); isolates each change's forward contribution
  - dca15: only DCA_DROP changes 1.25 -> 1.5 (secondary candidate)
  - reentry4: after closing a cycle, wait for a 4% pullback before re-entry
  - dca_vol_m1: volatility-scaled DCA amount; reduces tail risk, stays observational
  - overlay650t8_regime_v2: 650 top-up with an 8% trail on the common classifier; the
    only candidate that keeps buying while price trends away from the last sale
  - B_dcabrake_regime_v2: skip DCA in confirmed downtrends; observational
  - overlay_safe_combo: 350 top-up with a 6% trail (its TP 4% / spacing 0.25 are now live)

The live configuration is regime-aware and uses the 4h volatility trail, so replay only
accepts 240m bars: a --interval other than 240 exits without a snapshot.

Given the same OHLC, results are deterministic and reproducible. Closed forward bars are
stored locally so the anchored window continues growing beyond Kraken's 720-bar limit.
Run once for cron or with --loop. Never read or write live bot state.

  ./myenv/bin/python kraken/shadow_live.py                 # 240m snapshot, append JSONL
  ./myenv/bin/python kraken/shadow_live.py --loop 240      # rerun every 240 minutes
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from state_io import atomic_text_writer  # noqa: E402
from shadow_runtime import (  # noqa: E402
    load_shadow_environment, prepare_shadow_runtime,
)
from botcore import required_env  # noqa: E402
prepare_shadow_runtime(ROOT, HERE)

CONFIG_ENV = os.path.join(HERE, "config.env")
DEFAULT_ENV = os.path.join(HERE, ".env")
LOG_DIR = os.path.join(ROOT, "logs", "shadow_live")


def _load_runtime_config(env_path: str | None = None,
                         config_path: str | None = None) -> None:
    """Reproduce kraken_bot load order exactly: .env takes priority, config fills gaps."""
    env_path = env_path or os.environ.get("ENV_FILE", DEFAULT_ENV)
    config_path = config_path or os.path.join(os.path.dirname(env_path) or ".", "config.env")
    load_shadow_environment(env_path, config_path)


def _variants(interval: int):
    _load_runtime_config()
    from strategies import spot_dca as strat
    replace = dataclasses.replace
    base = strat.StratParams.from_env()
    variants = {
        "current": base,
        "pre0923": replace(
            base, takeprofit_pct=5.0, dca_spacing_growth_pct=0.0,
            tp_trail_adaptive=False, tp_regime_gate=False,
            tp_trail_profit_floor_pct=0.0, stop_loss_pct=12.5,
        ),
        "rev_tp5": replace(base, takeprofit_pct=5.0),
        "rev_spacing0": replace(base, dca_spacing_growth_pct=0.0),
        "rev_trail_fixed": replace(base, tp_trail_adaptive=False),
        "rev_gate_off": replace(base, tp_regime_gate=False),
        "rev_floor0": replace(base, tp_trail_profit_floor_pct=0.0),
        "rev_sl125": replace(base, stop_loss_pct=12.5),
        "dca15": replace(base, dca_drop_pct=1.5),
        "reentry4": replace(base, reentry_drop_pct=4.0),
        # DCA sizing reduces historical tail risk but fails the return/pairwise gate.
        "dca_vol_m1": replace(base, dca_vol_scale_k=-1.0, dca_vol_ref=2.0),
        # Versioned IDs prevent evidence from the former private classifier being mixed in.
        "overlay650t8_regime_v2": replace(
            base, trend_overlay=True, trend_topup=650.0, trend_trail_pct=8.0,
            trend_exit_break=False,
        ),
        # The DCA brake reduces historical tail risk but sacrifices returns.
        "B_dcabrake_regime_v2": replace(
            base, dca_trend_brake=True, dca_brake_min_pct=1.5,
        ),
        "overlay_safe_combo": replace(
            base, trend_overlay=True, trend_topup=350.0, trend_trail_pct=6.0,
            trend_exit_break=False,
        ),
    }
    return variants


def _replay_interval_error(interval: int) -> str | None:
    """Return why the live configuration cannot be replayed at this bar size, if it can't."""
    _load_runtime_config()
    from strategies import spot_dca as strat
    base = strat.StratParams.from_env()
    required = {int(base.trend_interval)} if (
        base.trend_overlay or base.dca_trend_brake
        or (base.tp_trend_hold and base.tp_regime_gate)) else set()
    if base.tp_trail_adaptive:
        required.add(int(base.tp_trail_vol_interval))
    if base.dca_vol_scale_k:
        required.add(int(base.dca_vol_interval))
    if required and required != {interval}:
        return (f"the live configuration needs {sorted(required)}-minute bars "
                f"(regime gate / volatility trail); --interval {interval} is not replayable")
    return None


def _fetch_with_ts(pair: str, interval: int):
    """Fetch candles like backtest.fetch_candles while retaining timestamps for anchoring.
    Return [(ts_sec, open, high, low, close), ...], excluding the forming final bar."""
    import json as _json
    import urllib.request
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (shadow)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = _json.loads(r.read())
    if data.get("error"):
        raise RuntimeError(", ".join(data["error"]))
    res = data.get("result", {})
    key = next((k for k in res if k != "last"), None)
    if not key:
        return []
    return [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
            for x in res[key][:-1]]


def _anchor_path(pair: str, interval: int) -> str:
    return os.path.join(LOG_DIR, f"{pair}_{interval}m.anchor")


def _history_path(pair: str, interval: int) -> str:
    return os.path.join(LOG_DIR, f"{pair}_{interval}m.ohlc.json")


def _get_anchor(pair: str, interval: int, default_ts: int) -> int:
    """On first run, anchor at the latest closed bar to begin forward testing now.
    Subsequent runs reuse it, so the window grows instead of sliding."""
    path = _anchor_path(pair, interval)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(default_ts))
    return default_ts


def _load_history(pair: str, interval: int) -> list[tuple[int, float, float, float, float]]:
    path = _history_path(pair, interval)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return [(int(row[0]), *(float(value) for value in row[1:])) for row in rows]


def _save_history(pair: str, interval: int, bars) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = _history_path(pair, interval)
    with atomic_text_writer(path) as fh:
        json.dump(bars, fh, separators=(",", ":"))
        fh.write("\n")


def _merge_forward_history(pair: str, interval: int, anchor: int, fetched):
    """Merge retained bars with the current fetch and detect missing history."""
    cached = [bar for bar in _load_history(pair, interval) if bar[0] >= anchor]
    fetched = [bar for bar in fetched if bar[0] >= anchor]

    if cached and fetched:
        expected_step = interval * 60
        if cached[-1][0] < fetched[0][0] - expected_step:
            raise RuntimeError(
                "the forward history has a gap larger than one interval; "
                "a complete anchored window cannot be reported"
            )

    merged_by_ts = {bar[0]: bar for bar in cached}
    merged_by_ts.update({bar[0]: bar for bar in fetched})
    merged = [merged_by_ts[ts] for ts in sorted(merged_by_ts)]
    if not merged or merged[0][0] != anchor:
        raise RuntimeError(
            f"the forward anchor {anchor} is no longer available in the local/Kraken history"
        )
    _save_history(pair, interval, merged)
    return merged


def _run_one(ohlc, params, interval, fee_pct, *, include_decision_trace=False):
    import replay as rp
    with contextlib.redirect_stdout(io.StringIO()):
        m = rp.run_replay(
            ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            include_decision_trace=include_decision_trace,
        )
    return m


def _decision_distance(reference: list[dict], candidate: list[dict]) -> int:
    """Count order events inserted, deleted, or replaced relative to current."""
    def fingerprint(event):
        return tuple(sorted(event.items()))

    matcher = difflib.SequenceMatcher(
        a=[fingerprint(event) for event in reference],
        b=[fingerprint(event) for event in candidate],
        autojunk=False,
    )
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def _eval_block(ohlc4, interval, fee_pct, *, include_decision_trace=False):
    """Run every variant over OHLC, returning None when fewer than two bars exist."""
    if len(ohlc4) < 2:
        return None
    rows = {}
    for name, params in _variants(interval).items():
        budget = float(params.effective_max_budget())
        m = _run_one(
            ohlc4, params, interval, fee_pct,
            include_decision_trace=include_decision_trace,
        )
        rows[name] = {
            "net_pct": round(m["net"] / budget * 100.0, 4),
            "total_pct": round(m["total"] / budget * 100.0, 4),  # includes open unrealized PnL
            "maxdd_pct": round(m.get("max_drawdown_pct") or 0.0, 4),
            "cycles": m.get("cycles", 0),
            "open_qty": m.get("open_qty", 0.0),
        }
        if include_decision_trace:
            rows[name]["decision_trace"] = m.get("decision_trace", [])
    if include_decision_trace:
        reference = rows["current"]["decision_trace"]
        for name, row in rows.items():
            row["decision_divergences"] = (
                0 if name == "current" else
                _decision_distance(reference, row["decision_trace"])
            )
    return rows


def snapshot(pair: str, interval: int, fee_pct: float, quiet: bool = False) -> dict:
    bars = _fetch_with_ts(pair, interval)
    if not bars:
        raise RuntimeError(f"fetch({pair},{interval}) returned empty")
    anchor = _get_anchor(pair, interval, bars[-1][0])
    full4 = [(o, h, l, c) for (_t, o, h, l, c) in bars]
    fwd_bars = _merge_forward_history(pair, interval, anchor, bars)
    fwd4 = [(o, h, l, c) for (_t, o, h, l, c) in fwd_bars]

    def _bh(seg):
        return round((seg[-1][3] / seg[0][3] - 1.0) * 100.0, 4) if len(seg) >= 2 else None

    snap = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pair": pair,
        "interval_min": interval,
        "anchor_ts": anchor,
        "last_close": bars[-1][4],
        "window": {"bars": len(full4), "buyhold_pct": _bh(full4),
                   "configs": _eval_block(full4, interval, fee_pct)},
        "forward": {"bars": len(fwd4), "buyhold_pct": _bh(fwd4),
                    "configs": _eval_block(
                        fwd4, interval, fee_pct, include_decision_trace=True,
                    )},
    }
    if not quiet:
        _print(snap)
    _append_jsonl(pair, interval, snap)
    return snap


def _print_block(title: str, blk: dict) -> None:
    r = blk.get("configs")
    bh = blk.get("buyhold_pct")
    bh_s = f"{bh:+.2f}%" if bh is not None else "n/a"
    print(f"  [{title}] {blk['bars']} bars  buy&hold {bh_s}")
    if not r:
        print("    (not enough bars — they are accumulating)")
        return
    cur = r["current"]["total_pct"]
    print(f"    {'config':<20} {'net%':>8} {'total%':>8} {'maxDD%':>8} "
          f"{'cycles':>8} {'Δdec':>6}  vs current total")
    for name, x in r.items():
        diff = "" if name == "current" else f"{x['total_pct'] - cur:+.2f}pp"
        divergences = x.get("decision_divergences")
        divergence_text = "-" if divergences is None else str(divergences)
        print(f"    {name:<20} {x['net_pct']:>8.2f} {x['total_pct']:>8.2f} "
              f"{x['maxdd_pct']:>8.2f} {x['cycles']:>8} "
              f"{divergence_text:>6}  {diff}")


def _print(snap: dict) -> None:
    print(f"[{snap['ts']}] {snap['pair']} {snap['interval_min']}m  last={snap['last_close']}")
    _print_block("FORWARD (from the anchor)", snap["forward"])
    _print_block("window (context, the complete window)", snap["window"])


def _append_jsonl(pair: str, interval: int, snap: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{pair}_{interval}m.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live shadow test: current vs pre-registered candidates (read-only)."
    )
    ap.add_argument("--interval", type=int, default=240, help="minutes per bar; the live configuration needs 240")
    ap.add_argument("--fee", type=float, default=0.26, help="fee per leg %%")
    ap.add_argument("--pair", default=None, help="KRAKEN_PAIR from .env/config.env by default")
    ap.add_argument("--loop", type=float, default=0.0, help="minutes between runs (0=single shot)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _load_runtime_config()
    pair = args.pair or required_env("KRAKEN_PAIR")
    reason = _replay_interval_error(args.interval)
    if reason:
        print(f"[shadow_live] skipped: {reason}")
        return 0

    while True:
        try:
            snapshot(pair, args.interval, args.fee, quiet=args.quiet)
        except Exception as e:  # in loop mode, one failed fetch does not stop monitoring
            print(f"[shadow_live] error: {e}", file=sys.stderr)
            if args.loop <= 0:
                return 1
        if args.loop <= 0:
            return 0
        time.sleep(args.loop * 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
