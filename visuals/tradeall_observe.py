#!/usr/bin/env python3
"""Separate observational monitor for the tradeall trend model.

This module neither imports nor starts tradeall.py or bapi_placeorder.py. It only
reads pipe-delimited logger files and cache_short_trend.json.

Live mode (default) runs every two seconds by default:
  1. Sample the current price into logger/tradeall_price_samples_YYYY-MM-DD.log.
  2. Read trend-start decisions and tradeall order outcomes.
  3. Reconstruct trend regions from one transition to the next.
  4. Draw live/day/week charts with price, state background, and order markers.
  5. Write a static, auto-refreshing tradeall_live.html with a day/week toggle.

Manual live invocation:
    ./tradeall_observe.py [--symbols BTCUSDC,TAOUSDC] [--interval 5]
    Then open tradeall_live.html locally or through ngrok.

Backtest mode renders an offline/backtests/tradeall.py run (A5):
    ./tradeall_observe.py --backtest-dir logger/backtest/<run_id> --symbols BTCUSDC
It performs no live sampling, reads flat files in that directory, uses the full
replayed interval so far, and writes the PNG into the same directory.
"""
import argparse
import ctypes
import gc
import json
from math import ceil
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from active_intents import build_active_intent_index
from botcore import (
    lru_dict_put,
    normalize_trend_direction,
    require_pos_float,
    safe_float,
    safe_int,
    safe_pos_float,
    safe_pos_int,
    sanitize_pipe_field,
    split_csv,
)
from state_io import append_text_line, atomic_file_path, atomic_write_text, read_json_dict, read_tail_jsonl_dicts
from providers.base import normalize_side
from intelligence.internal.state.volatility import get_kalman_state_path
from intelligence.macro_sentiment_ssot import lookup_symbol_cache_items
from cachepaths import (
    CACHE_SHORT_TREND_FILENAME,
    cache_24_price_filename,
    cache_365_price_filename,
    cache_path,
    cache_sparse_price_filename,
    cache_symbol_variants,
    get_cachedb_dir,
    resolve_cachedb_filepath,
)
from instrument_registry import symbols_for
LOGGER_DIR = os.path.join(ROOT, "logger")
CACHE_TREND_PATH = cache_path(CACHE_SHORT_TREND_FILENAME)

# Dedicated live-mode output directory (PNGs and HTML), deliberately separated
# from the repository. Unlike ROOT, it contains only generated charts and never
# .env files, keys, or cachedb, making it the only safe directory to serve.
LIVE_OUT_DIR = os.path.join(ROOT, "tradeall_live")

PRICE_SAMPLES_PREFIX = "tradeall_price_samples_"
DECISIONS_PREFIX = "tradeall_decisions_"
OUTCOMES_PREFIX = "order_outcomes_"
SHADOW_PREFIX = "tradeall_shadow_"
SHADOW_COLOR = "#8250df"   # violet — shadow signals (Kalman), distinct from green/red

STATE_COLORS = {"UP": "#1a7f37", "DOWN": "#cf222e", "HOLD": "#8c8c8c"}
DAY_SECONDS = 24 * 3600
WEEK_SECONDS = 7 * DAY_SECONDS
MEMORY_TRIM_SECONDS = 300


_PIPE_LOG_CACHE_MAX_FILES = 32
_PIPE_LOG_CACHE = OrderedDict()
# path -> (bytes consumed, ncols, rows, device, inode, observed size, mtime_ns)


def _read_pipe_log(path, ncols):
    """Read a pipe-delimited file, ignoring malformed rows from interrupted writes.

    The incremental cache remembers the byte offset for each append-only daily
    file and parses only new data. A closed day's file therefore incurs no I/O
    after its first read. Binary seek avoids text-mode encoding offset issues.
    """
    try:
        stream = open(path, "rb")
    except OSError:
        # Log rotation deletes old files. Return [], while retaining that file's
        # parsed rows for the process lifetime.
        _PIPE_LOG_CACHE.pop(path, None)
        return []
    with stream:
        stat = os.fstat(stream.fileno())
        cached = _PIPE_LOG_CACHE.get(path)
        same_file = bool(
            cached
            and cached[1] == ncols
            and cached[3:5] == (stat.st_dev, stat.st_ino)
        )
        unchanged = bool(
            same_file
            and cached[5] == stat.st_size
            and cached[6] == stat.st_mtime_ns
        )
        if unchanged:
            _PIPE_LOG_CACHE.move_to_end(path)
            return cached[2]                   # unchanged -> zero I/O

        # Re-read same-size rewrites and rotations to a different inode.
        same_size_rewrite = bool(same_file and cached[5] == stat.st_size)
        if same_file and not same_size_rewrite and stat.st_size >= cached[0]:
            offset, rows = cached[0], list(cached[2])
        else:
            offset, rows = 0, []               # New or truncated file -> start over
        try:
            stream.seek(offset)
            chunk = stream.read()
            read_stat = os.fstat(stream.fileno())
        except OSError:
            return rows
    consumed = offset
    for raw in chunk.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            break                               # incomplete line (write in progress) — retry next cycle
        consumed += len(raw)
        parts = raw.decode("utf-8", errors="replace").rstrip("\n").split("|")
        if len(parts) < ncols:
            continue
        rows.append(parts)
    lru_dict_put(
        _PIPE_LOG_CACHE,
        path,
        (
            consumed,
            ncols,
            rows,
            read_stat.st_dev,
            read_stat.st_ino,
            read_stat.st_size,
            read_stat.st_mtime_ns,
        ),
        _PIPE_LOG_CACHE_MAX_FILES,
    )
    return rows


def _log_dates(days_back):
    n_days = safe_pos_int(days_back, default=None)
    if n_days is None:
        return []
    today = datetime.now().date()
    return [today - timedelta(days=i) for i in range(n_days)]


# -- Step B: independent price sampling that does not touch tradeall. ----------

_LAST_SAMPLED_TS = {}


def sample_current_prices(symbols):
    if not os.path.exists(CACHE_TREND_PATH):
        return
    try:
        snapshot = read_json_dict(CACHE_TREND_PATH, recover_previous=True)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[tradeall_observe] error reading cache_short_trend.json: {e}")
        return
    items_map = snapshot.get("items") if isinstance(snapshot.get("items"), dict) else snapshot

    pending = []
    for symbol in symbols:
        entry = _lookup_symbol_dict(items_map, symbol)
        if not isinstance(entry, dict) or "current_price" not in entry:
            continue
        raw_ts = entry.get("ts", time.time())
        raw_px = entry["current_price"]
        sample_ts = safe_pos_float(raw_ts, default=None)
        px_val = safe_pos_float(raw_px, default=None)
        if sample_ts is None or px_val is None:
            continue
        # The loop runs every two seconds, but the source may publish less often.
        # Avoid recording the same tick repeatedly and inflating the log and RAM cache.
        if sample_ts <= _LAST_SAMPLED_TS.get(symbol, float("-inf")):
            continue
        pending.append((symbol, sample_ts, raw_px))
    if not pending:
        return

    os.makedirs(LOGGER_DIR, exist_ok=True)
    path = os.path.join(LOGGER_DIR, f"{PRICE_SAMPLES_PREFIX}{datetime.now().date().isoformat()}.log")
    try:
        lines = []
        for symbol, sample_ts, current_price in pending:
            # Use the snapshot timestamp, not our clock. If the source freezes,
            # a current timestamp on stale data would create a false jagged block.
            lines.append(f"{sample_ts}|{sanitize_pipe_field(symbol)}|{current_price}")
        append_text_line(path, "\n".join(lines))
        for symbol, sample_ts, _ in pending:
            _LAST_SAMPLED_TS[symbol] = sample_ts
    except OSError as e:
        print(f"[tradeall_observe] error writing the price samples: {e}")


# -- Load logs (steps A / A2). -------------------------------------------------

def _parse_price_sample_rows(rows, symbol, *, require_pos_ts: bool = False):
    ts_list, px_list = [], []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        ts, sym_, price = row[:3]
        if sym_ != symbol:
            continue
        t_val = safe_pos_float(ts, default=None) if require_pos_ts else safe_float(ts, default=None)
        p_val = safe_pos_float(price, default=None)
        if t_val is not None and p_val is not None:
            ts_list.append(t_val)
            px_list.append(p_val)
    return ts_list, px_list


def _read_daily_pipe_rows(prefix, ncols, days_back):
    rows = []
    for d in reversed(_log_dates(days_back)):
        rows.extend(_read_pipe_log(os.path.join(LOGGER_DIR, f"{prefix}{d.isoformat()}.log"), ncols))
    return rows


def load_price_samples(symbol, days_back):
    return _parse_price_sample_rows(_read_daily_pipe_rows(PRICE_SAMPLES_PREFIX, 3, days_back), symbol, require_pos_ts=False)


def _load_cachedb_price_entries(filename, symbol, max_bytes=4 * 1024 * 1024):
    """Read Cache24 ``[[ts_ms, price], ...]`` entries from cachedb.

    Incremental long-term JSONL files may grow to hundreds of megabytes, so the
    weekly caller reads only their ``max_bytes`` tail. The legacy JSON format
    used by the live 24-hour cache remains supported unchanged.
    """
    path = str(resolve_cachedb_filepath(os.path.join(get_cachedb_dir(), filename)))
    if not os.path.exists(path):
        return []
    if filename.endswith(".jsonl"):
        sym_variants = set(cache_symbol_variants(symbol, include_original=True))
        return [
            rec["i"]
            for rec in read_tail_jsonl_dicts(path, limit=None, max_bytes=max_bytes)
            if rec.get("s") in sym_variants and "i" in rec
        ]
    data = read_json_dict(path, default={}, recover_previous=True)
    return lookup_symbol_cache_items(data, symbol, allow_base_fallback=False, require_list=True, unwrap_items=True) or []


def _ingest_cachedb_points(points: dict, entries) -> None:
    for entry in entries or ():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        t_ms = safe_pos_float(entry[0], default=None)
        p_val = safe_pos_float(entry[1], default=None)
        if t_ms is not None and p_val is not None:
            points[int(t_ms / 1000.0)] = p_val


def load_price_series_live(symbol, days_back, include_history=True):
    """Merge and sort all existing sources into an immediately populated chart.

    Sources comprise sparse long-term history, the dense long-term archive, this
    monitor's samples, and the fleet's dense live 24-hour cache. Dense points win
    timestamp collisions. ``include_history=False`` skips sparse history for
    windows already covered by dense caches, saving parsing on each cycle.
    """
    points = {}
    if include_history:
        _ingest_cachedb_points(points, _load_cachedb_price_entries(cache_sparse_price_filename(symbol), symbol))
        _ingest_cachedb_points(points, _load_cachedb_price_entries(cache_365_price_filename(symbol), symbol))
    own_ts, own_px = load_price_samples(symbol, days_back)
    for t, p in zip(own_ts, own_px):
        t_val = safe_pos_float(t, default=None)
        p_val = safe_pos_float(p, default=None)
        if t_val is not None and p_val is not None:
            points[int(t_val)] = p_val
    _ingest_cachedb_points(points, _load_cachedb_price_entries(cache_24_price_filename(symbol), symbol))
    ts_sorted = sorted(points)
    return ts_sorted, [points[t] for t in ts_sorted]


def _parse_trend_start_rows(rows, symbol):
    events = []
    for row in rows:
        ts, sym_, event, state, old_state, price, prev_confirm = row[:7]
        if sym_ != symbol or event != "trend_start":
            continue
        t_val = safe_pos_float(ts, default=None)
        p_val = safe_pos_float(price, default=None)
        if t_val is None or p_val is None:
            continue
        events.append({"ts": t_val, "state": state, "old_state": old_state,
                        "price": p_val, "prev_confirm_count": prev_confirm})
    events.sort(key=lambda e: e["ts"])
    return events


def load_trend_starts(symbol, days_back):
    return _parse_trend_start_rows(_read_daily_pipe_rows(DECISIONS_PREFIX, 7, days_back), symbol)


def _parse_order_event_rows(rows, symbol, *, filter_caller: bool = True):
    events = []
    for row in rows:
        ts, sym_, side, price, qty, outcome, refuse_reason, caller, motivation = row[:9]
        is_ta_order = caller == "tradeall.py" or (
            caller == "market_api.py"
            and bool(motivation)
            and any(
                motivation.startswith(prefix)
                for prefix in (
                    "trend_", "kalman_", "consistent_", "force_", "test_", "surge_",
                    "small_window_", "accel_", "slope<=", "slope>=",
                )
            )
        )
        if sym_ != symbol or (filter_caller and not is_ta_order):
            continue
        t_val = safe_pos_float(ts, default=None)
        p_val = safe_pos_float(price, default=None)
        if t_val is None or p_val is None:
            continue
        events.append({
            "ts": t_val, "side": normalize_side(side) or side, "price": p_val,
            "outcome": outcome, "reason": motivation or refuse_reason,
        })
    events.sort(key=lambda e: e["ts"])
    return events


def load_order_events(symbol, days_back):
    """Load only tradeall.py events from the fleet-wide outcome log."""
    return _parse_order_event_rows(_read_daily_pipe_rows(OUTCOMES_PREFIX, 9, days_back), symbol, filter_caller=True)


def _parse_shadow_rows(rows, symbol):
    """Parse shadow rows: ts|symbol|signal|event|state|old_state|price|vel|vel_std."""
    events = []
    for row in rows:
        ts, sym_, signal, event, state, old_state, price, vel, vel_std = row[:9]
        if sym_ != symbol or event != "trend_start":
            continue
        t_val = safe_pos_float(ts, default=None)
        p_val = safe_pos_float(price, default=None)
        st_val = safe_int(state, default=None)
        ost_val = safe_int(old_state, default=None)
        if st_val is None or ost_val is None or t_val is None or p_val is None:
            continue
        events.append({"ts": t_val, "signal": signal, "state": st_val,
                        "old_state": ost_val, "price": p_val,
                        "vel": vel, "vel_std": vel_std})
    events.sort(key=lambda e: e["ts"])
    return events


def load_shadow_events(symbol, days_back):
    return _parse_shadow_rows(_read_daily_pipe_rows(SHADOW_PREFIX, 9, days_back), symbol)


def build_trend_regions(events, window_start, window_end):
    """Build state regions from each trend start to the next (plan steps A/C)."""
    if not events:
        return []
    ws = safe_float(window_start, 0.0)
    we = safe_float(window_end, 0.0)
    regions = []
    for i, ev in enumerate(events):
        if not isinstance(ev, dict) or "ts" not in ev or "state" not in ev:
            continue
        ev_ts = safe_float(ev.get("ts"), 0.0)
        next_ts = safe_float(events[i + 1].get("ts"), we) if (i + 1 < len(events) and isinstance(events[i + 1], dict)) else we
        start, end = max(ev_ts, ws), min(next_ts, we)
        if end > start:
            regions.append((start, end, ev["state"]))
    return regions


# -- Load flat, non-rotating backtest logs. ------------------------------------

def load_backtest_price_samples(directory, symbol):
    return _parse_price_sample_rows(
        _read_pipe_log(os.path.join(directory, "tradeall_price_samples.log"), 3),
        symbol,
        require_pos_ts=True,
    )


def load_backtest_trend_starts(directory, symbol):
    return _parse_trend_start_rows(
        _read_pipe_log(os.path.join(directory, "tradeall_decisions.log"), 7),
        symbol,
    )


def load_backtest_order_events(directory, symbol):
    """Load one backtest run; its sole caller needs no caller filtering."""
    return _parse_order_event_rows(
        _read_pipe_log(os.path.join(directory, "order_outcomes.log"), 9),
        symbol,
        filter_caller=False,
    )


# -- Rendering (step C). -------------------------------------------------------

def _close_new_figures(function):
    """Ensure cleanup even when rendering fails before saving."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        existing = set(plt.get_fignums())
        try:
            return function(*args, **kwargs)
        finally:
            for figure_number in set(plt.get_fignums()) - existing:
                plt.close(figure_number)

    return wrapped


def _save_figure_atomic(fig, out_path, **savefig_kwargs):
    """Publish a PNG only after rendering completes successfully."""
    with atomic_file_path(out_path, prefix=".tradeall_observe_", suffix=".png") as temp_path:
        fig.savefig(temp_path, format="png", **savefig_kwargs)


def _release_unused_memory():
    """Release Python cycles and, on glibc, return large arenas to the OS.

    This best-effort operation runs only after infrequent weekly rendering. It
    is not required for correctness and remains portable without malloc_trim.
    """
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return False
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return bool(malloc_trim(0))


def _plot_order_markers(ax, events, *, filled, max_annotated=25):
    """Plot orders in batches instead of one PathCollection per order.

    Weekly charts may contain thousands of refusals. One scatter call per row
    created thousands of Matplotlib objects and pushed peak RAM above 1 GiB;
    two BUY/SELL batches preserve the same visual markers.
    """
    sides = list(dict.fromkeys(["BUY", "SELL", *(e["side"] for e in events)]))
    annotate = len(events) <= max_annotated
    for side in sides:
        group = [event for event in events if event["side"] == side]
        if not group:
            continue
        marker = "^" if side == "BUY" else "v"
        color = STATE_COLORS["UP"] if side == "BUY" else STATE_COLORS["DOWN"]
        scatter_args = {
            "marker": marker,
            "s": 90,
            "zorder": 5,
        }
        if filled:
            scatter_args["color"] = color
        else:
            scatter_args.update(
                facecolors="none", edgecolors=color, linewidths=1.5
            )
        ax.scatter(
            [datetime.fromtimestamp(event["ts"]) for event in group],
            [event["price"] for event in group],
            **scatter_args,
        )
        if annotate:
            y_offset = 6 if filled else -10
            for event in group:
                ax.annotate(
                    event["reason"],
                    (datetime.fromtimestamp(event["ts"]), event["price"]),
                    fontsize=7,
                    xytext=(4, y_offset),
                    textcoords="offset points",
                    color=color,
                )


@_close_new_figures
def render_chart(symbol, window_label, window_start, window_end,
                  price_ts, price_vals, trend_events, order_events, out_path,
                  state_text=None, shadow_events=None):
    """Draw a chart from an already loaded live or backtest data set.

    ``state_text`` is an optional current-analysis box. ``shadow_events`` adds
    violet diamonds and dotted lines for visual comparison with the main model.
    """
    vis_ts, vis_px = [], []
    for t, p in zip(price_ts, price_vals):
        if window_start <= t <= window_end:
            vis_ts.append(t)
            vis_px.append(p)

    fig, ax = plt.subplots(figsize=(11, 5))

    for start, end, state in build_trend_regions(trend_events, window_start, window_end):
        ax.axvspan(datetime.fromtimestamp(start), datetime.fromtimestamp(end),
                   color=STATE_COLORS.get(state, "#dddddd"), alpha=0.15, lw=0)

    # Mark the exact time of every trend transition, not only its background region.
    # Put UP labels above and DOWN labels below price. Progressively stagger nearby
    # same-direction events to prevent overlapping labels.
    visible_trend_starts = [e for e in trend_events if window_start <= e["ts"] <= window_end]
    cluster_gap = max((window_end - window_start) * 0.02, 1.0)   # 2% of the visible window.
    last_ts_by_state = {}
    stagger_by_state = {}
    for e in visible_trend_starts:
        state = e["state"]
        prev_ts = last_ts_by_state.get(state)
        stagger_by_state[state] = stagger_by_state.get(state, -1) + 1 \
            if prev_ts is not None and (e["ts"] - prev_ts) < cluster_gap else 0
        last_ts_by_state[state] = e["ts"]
        level = stagger_by_state[state]

        color = STATE_COLORS.get(state, "#888888")
        t = datetime.fromtimestamp(e["ts"])
        ax.axvline(t, color=color, ls="--", lw=1, alpha=0.6, zorder=3)
        ax.scatter(t, e["price"], marker="o", color=color, s=40, zorder=6,
                   edgecolors="white", linewidths=0.8)
        label = f"{e['old_state']}→{e['state']}"
        if e.get("prev_confirm_count"):
            label += f" ({e['prev_confirm_count']})"
        step = 15 * level
        y_offset = 16 + step if state == "UP" else -20 - step
        va = "bottom" if state == "UP" else "top"
        ax.annotate(label, (t, e["price"]), fontsize=5.5, xytext=(3, y_offset),
                    textcoords="offset points", color=color, va=va, rotation=0)

    if vis_ts:
        # Downsample dense views: above ~6,000 points on ~1,100 pixels, a daily
        # one-second series becomes an unreadable solid band.
        if len(vis_ts) > 6000:
            stride = len(vis_ts) // 6000 + 1
            vis_ts = vis_ts[::stride]
            vis_px = vis_px[::stride]
        ax.plot([datetime.fromtimestamp(t) for t in vis_ts], vis_px,
                color="#1f6feb", lw=0.8, label="price")
    else:
        ax.text(0.5, 0.5, "No price samples yet\n(let the monitor run for a while)",
                ha="center", va="center", transform=ax.transAxes, color="#888888")

    # Shadow Kalman transitions use violet diamonds and dotted lines so they are
    # visually distinct from the model's trend starts.
    shadow_map = {1: "K:UP", -1: "K:DOWN", 0: "K:FLAT"}
    for e in (shadow_events or []):
        if not (window_start <= e["ts"] <= window_end):
            continue
        t = datetime.fromtimestamp(e["ts"])
        ax.axvline(t, color=SHADOW_COLOR, ls=":", lw=1, alpha=0.7, zorder=3)
        ax.scatter(t, e["price"], marker="D", color=SHADOW_COLOR, s=32, zorder=6,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(shadow_map.get(e["state"], "?"), (t, e["price"]), fontsize=5.5,
                    xytext=(3, 8), textcoords="offset points", color=SHADOW_COLOR)

    visible_orders = [e for e in order_events if window_start <= e["ts"] <= window_end]
    # ``accepted`` is submission truth, never fill truth. Continue reading legacy
    # ``executed`` rows without perpetuating that misleading label in new output.
    accepted = [
        e for e in visible_orders
        if e["outcome"] in {"accepted", "executed"}
    ]
    refused = [e for e in visible_orders if e["outcome"] == "refused"]

    # Limit marker annotations when attempts are dense. Filled marker styling below
    # represents provider acceptance, not a confirmed venue fill.
    MAX_ANNOTATED = 25

    _plot_order_markers(ax, accepted, filled=True, max_annotated=MAX_ANNOTATED)
    _plot_order_markers(ax, refused, filled=False, max_annotated=MAX_ANNOTATED)

    summary = (
        f"BUY  accepted: {sum(1 for e in accepted if e['side'] == 'BUY')}   "
        f"refused: {sum(1 for e in refused if e['side'] == 'BUY')}\n"
        f"SELL accepted: {sum(1 for e in accepted if e['side'] == 'SELL')}   "
        f"refused: {sum(1 for e in refused if e['side'] == 'SELL')}"
    )
    ax.text(0.01, 0.98, summary, transform=ax.transAxes, va="top", fontsize=9,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))

    # Use the requested X window, not merely the data range, so day and week views
    # remain distinct when all available data fits within 24 hours.
    if state_text:
        ax.text(0.99, 0.98, state_text, transform=ax.transAxes, va="top", ha="right",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4a017", alpha=0.95))

    if window_end <= window_start:
        window_end = window_start + 60.0
    ax.set_xlim(datetime.fromtimestamp(window_start), datetime.fromtimestamp(window_end))
    ax.set_title(f"{symbol} — {window_label}  (updated {datetime.now().strftime('%H:%M:%S')})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    # autofmt_xdate plus tight_layout measured text on every render; profiling
    # attributed ~96 ms of ~360 ms per call to tight_layout. The structure is
    # stable, so use margins calibrated from a reference run and apply the
    # default 30-degree label rotation without remeasurement.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower left", fontsize=8)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.16)
    try:
        _save_figure_atomic(fig, out_path, dpi=110)
    finally:
        plt.close(fig)


def _get_active_summary():
    """Safely fetch current active intent summary without raising."""
    try:
        return build_active_intent_index(ROOT).get("summary")
    except Exception:
        return None


def format_state_text(entry, header, pending=None, exposure=None):
    if not isinstance(entry, dict):
        entry = {}
    trend_map = {1: "UP", -1: "DOWN", 0: "FLAT"}

    def _t_label(val):
        if val is None or isinstance(val, bool):
            return "?"
        iv = safe_int(val, default=None)
        if iv in trend_map:
            return trend_map[iv]
        if isinstance(val, str):
            d = normalize_trend_direction(val, default=None)
            if d:
                return d.upper()
            if val.strip().upper() in ("FLAT", "HOLD", "NEUTRAL"):
                return "FLAT"
        return "?"

    text = (
        f"{header}\n"
        f"price:     {entry.get('current_price', '?')}\n"
        f"trend:     {_t_label(entry.get('final_trend'))}\n"
        f"grad rec:  {safe_float(entry.get('gradient_recent')):+.4f}\n"
        f"slope short:{safe_float(entry.get('slope_small')):+.3f}\n"
        f"slope long: {safe_float(entry.get('slope_big')):+.3f}\n"
        f"epsilon:   {safe_float(entry.get('epsilon')):.4f}"
    )
    fast_trend = entry.get("trend_fast")
    if fast_trend is not None:
        text += f"\nfast:      {_t_label(fast_trend)} (grad: {safe_float(entry.get('gradient_recent_fast')):+.4f})"
    # Add shadow rows only when their keys exist in the snapshot.
    if entry.get("kalman_trend") is not None:
        text += (f"\nkalman:    {_t_label(entry.get('kalman_trend'))} "
                 f"(v={safe_float(entry.get('kalman_vel')):+.3f}%/min ±{safe_float(entry.get('kalman_vel_std')):.3f})")
    v1h_f = safe_float(entry.get("vol_1h_pct"), default=None)
    if v1h_f is not None:
        text += f"\nvol 1h:    {v1h_f:.2f}%"
        if entry.get("adapt_reentry_pct") is not None or entry.get("adapt_dca_pct") is not None:
            text += (f" → re:{entry.get('adapt_reentry_pct', '?')}% "
                     f"dca:{entry.get('adapt_dca_pct', '?')}%")
    if pending:
        text += f"\npending:   {pending}"
    if exposure:
        text += f"\nposition:  {exposure}"
    return text


def _lookup_symbol_dict(mapping, symbol):
    return lookup_symbol_cache_items(mapping, symbol, allow_base_fallback=False, require_dict=True, unwrap_items=True)


def build_backtest_state_text(directory, symbol):
    """Load simulated analysis state written by offline/backtests/tradeall.py."""
    path = os.path.join(directory, "analysis_state.json")
    entry = _lookup_symbol_dict(read_json_dict(path, default={}, recover_previous=True), symbol)
    if not isinstance(entry, dict) or not entry:
        return None
    try:
        sim_time = datetime.fromtimestamp(safe_float(entry.get("ts"), 0.0)).strftime("%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        sim_time = "unknown"
    return format_state_text(entry, f"SIMULATED ANALYSIS ({sim_time})")


@_close_new_figures
def render_state_image(state_text, out_path):
    """Render the state box separately for the HTML chart's hover display."""
    fig = plt.figure(figsize=(4.8, 3.2))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.96, state_text, va="top", ha="left", fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#fffbe6", ec="#d4a017", alpha=0.97))
    try:
        _save_figure_atomic(fig, out_path, dpi=110, transparent=True)
    finally:
        plt.close(fig)


def render_symbol_charts_live(symbol, chart_specs, *, window_end=None):
    """Render one or more windows from one data snapshot.

    ``chart_specs`` contains ``(label, seconds, png_path)`` tuples. Sharing a
    snapshot avoids rebuilding the same logs and caches for each due window.
    """
    if not chart_specs:
        return
    we_val = safe_pos_float(window_end, default=0.0)
    window_end = we_val or time.time()
    max_window = max(window_seconds for _, window_seconds, _ in chart_specs)
    days_back = 9 if max_window > DAY_SECONDS else 2
    include_history = max_window > DAY_SECONDS
    price_ts, price_vals = load_price_series_live(
        symbol, days_back, include_history=include_history
    )
    trend_events = load_trend_starts(symbol, days_back)
    order_events = load_order_events(symbol, days_back)
    shadow_events = load_shadow_events(symbol, days_back)
    for window_label, window_seconds, out_path in chart_specs:
        render_chart(
            symbol,
            window_label,
            window_end - window_seconds,
            window_end,
            price_ts,
            price_vals,
            trend_events,
            order_events,
            out_path,
            shadow_events=shadow_events,
        )


# -- Static, serverless HTML with day/week toggle and auto-refresh. ------------

_BASE_CHART_CSS = """body { font-family: -apple-system, sans-serif; background:#111; color:#eee; margin: 16px; }
img { max-width: 100%; border: 1px solid #333; border-radius: 4px; }
.chart { margin-bottom: 24px; }
.chart-wrap { position: relative; display: inline-block; max-width: 100%; }
.state-overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                  opacity: 0; transition: opacity .15s; pointer-events: none;
                  max-width: 45%; border: none; }
.chart-wrap:hover .state-overlay { opacity: 1; }"""


def write_html(symbols, live_minutes=60, active_summary=None):
    live_minutes = safe_pos_float(live_minutes, default=60.0)
    blocks = [
        f'\n  <div class="chart">\n    <h3>{s}</h3>\n    <div class="chart-wrap">\n'
        f'      <img class="view-live" src="tradeall_live_{s}_live.png" alt="{s} live">\n'
        f'      <img class="view-day" src="tradeall_live_{s}_day.png" alt="{s} day" style="display:none">\n'
        f'      <img class="view-week" src="tradeall_live_{s}_week.png" alt="{s} week" style="display:none">\n'
        f'      <img class="state-overlay" src="tradeall_live_{s}_state.png" alt="{s} state">\n'
        f'    </div>\n  </div>'
        for s in symbols
    ]

    summary_html = ""
    if isinstance(active_summary, dict) and active_summary:
        notional = active_summary.get("pending_notional") if isinstance(active_summary.get("pending_notional"), dict) else {}
        exposures = active_summary.get("net_exposures") if isinstance(active_summary.get("net_exposures"), dict) else {}
        summary_html = f'''
<div class="summary-bar">
  <span><strong>Fleet Pending BUY:</strong> ${safe_float(notional.get("total_buy"), 0.0):.2f}</span>
  <span><strong>Fleet Pending SELL:</strong> ${safe_float(notional.get("total_sell"), 0.0):.2f}</span>
  <span><strong>Open Cost Basis:</strong> ${safe_float(exposures.get("total_cost"), 0.0):.2f}</span>
</div>'''

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tradeall — live monitor</title>
<style>
{_BASE_CHART_CSS}
button {{ margin: 4px 6px 14px 0; padding: 6px 16px; border-radius: 4px; border: 1px solid #444;
         background:#222; color:#eee; cursor:pointer; }}
button.active {{ background:#1f6feb; border-color:#1f6feb; }}
.summary-bar {{ background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 10px 16px;
                margin-bottom: 18px; display: flex; flex-wrap: wrap; gap: 20px; font-family: monospace; font-size: 13px; }}
.summary-bar span strong {{ color: #58a6ff; }}
</style></head>
<body>
<h2>tradeall — live monitor</h2>
{summary_html}
<button id="btn-live" class="active" onclick="show('live')">Live ({live_minutes:.0f} min)</button>
<button id="btn-day" onclick="show('day')">Day</button>
<button id="btn-week" onclick="show('week')">Week</button>
{"".join(blocks)}
<script>
function bust() {{
  document.querySelectorAll('img').forEach(function(img) {{
    var base = img.src.split('?')[0];
    img.src = base + '?t=' + Date.now();
  }});
}}
function show(which) {{
  ['live', 'day', 'week'].forEach(function(v) {{
    document.querySelectorAll('.view-' + v).forEach(function(i) {{ i.style.display = which === v ? '' : 'none'; }});
    document.getElementById('btn-' + v).classList.toggle('active', which === v);
  }});
}}
setInterval(bust, 2500);
</script>
</body></html>"""
    atomic_write_text(os.path.join(LIVE_OUT_DIR, "tradeall_live.html"), html)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--symbols", default=None,
                         help="explicit override; otherwise all enabled Binance instruments")
    parser.add_argument("--interval", type=float, default=2.0,
                         help="seconds between cycles (live plus the analysis state; default 2)")
    parser.add_argument("--live-minutes", type=float, default=60.0,
                         help="the Live tab's window, in minutes (default 60; TAO moves rarely "
                              "— at 30 min the window shows more emptiness than signal)")
    parser.add_argument("--day-refresh", type=float, default=30.0,
                         help="how often, in seconds, the DAY chart is redrawn (default 30)")
    parser.add_argument("--week-refresh", type=float, default=300.0,
                         help="how often, in seconds, the WEEK chart is redrawn (default 300)")
    parser.add_argument("--backtest-dir", default=None,
                         help="if given: BACKTEST mode — it renders the folder of an "
                              "offline/backtests/tradeall.py run instead of running live")
    parser.add_argument("--frame-hours", type=float, default=None,
                         help="with --backtest-dir: instead of ONE dense chart covering the whole interval, "
                              "generate a SERIES of STATIC frames (images), one per N hours. "
                              "Generate once and exit (does not loop).")
    parser.add_argument("--window-hours", type=float, default=None,
                         help="with --backtest-dir (without --frame-hours): a SLIDING window of N hours, "
                              "anchored to the current simulated clock — it runs in a loop (like live), "
                              "sliding as offline/backtests/tradeall.py writes new data in parallel; "
                              "the events appear on the frame exactly when the backtester reaches them)")
    args = parser.parse_args()
    interval_sec = require_pos_float(args.interval, name="--interval")
    live_minutes = require_pos_float(args.live_minutes, name="--live-minutes")
    day_refresh = require_pos_float(args.day_refresh, name="--day-refresh")
    week_refresh = require_pos_float(args.week_refresh, name="--week-refresh")
    frame_hours = (
        require_pos_float(args.frame_hours, name="--frame-hours")
        if args.frame_hours is not None
        else None
    )
    window_hours = (
        require_pos_float(args.window_hours, name="--window-hours")
        if args.window_hours is not None
        else None
    )
    symbols = (
        symbols_for("binance")
        if args.symbols is None
        else list(split_csv(args.symbols))
    )

    if args.backtest_dir and frame_hours is not None:
        directory = os.path.abspath(args.backtest_dir)
        chunk_sec = frame_hours * 3600
        for symbol in symbols:
            price_ts, price_vals = load_backtest_price_samples(directory, symbol)
            trend_events = load_backtest_trend_starts(directory, symbol)
            order_events = load_backtest_order_events(directory, symbol)
            all_ts = price_ts + [e["ts"] for e in trend_events] + [e["ts"] for e in order_events]
            if not all_ts:
                print(f"[tradeall_observe] {symbol}: no data for the frames yet")
                paths = []
            else:
                range_start, range_end = min(all_ts), max(all_ts)
                n_chunks = max(1, int(ceil((range_end - range_start) / chunk_sec))) if range_end > range_start else 1
                paths = []
                for i in range(n_chunks):
                    c_start = range_start + i * chunk_sec
                    c_end = min(c_start + chunk_sec, range_end)
                    try:
                        start_dt = datetime.fromtimestamp(c_start)
                        end_dt = datetime.fromtimestamp(c_end)
                        start_label = start_dt.strftime('%Y-%m-%d %H:%M')
                        end_label = end_dt.strftime('%Y-%m-%d %H:%M')
                        start_tag = start_dt.strftime('%Y%m%d_%H%M')
                    except (ValueError, OSError, OverflowError):
                        start_label = str(int(c_start))
                        end_label = str(int(c_end))
                        start_tag = str(int(c_start))
                    label = f"{start_label} → {end_label}  (frame {i+1}/{n_chunks})"
                    path = os.path.join(directory, f"tradeall_live_{symbol}_frame{i+1:03d}_{start_tag}.png")
                    render_chart(symbol, label, c_start, c_end, price_ts, price_vals, trend_events, order_events, path)
                    paths.append(path)
            print(f"[tradeall_observe] {symbol}: {len(paths)} frames generated in {directory}")
            for p in paths:
                print(f"    {p}")
        return

    if args.backtest_dir:
        directory = os.path.abspath(args.backtest_dir)
        blocks = "".join(
            f'<div class="chart"><h3>{s}</h3><div class="chart-wrap">'
            f'<img src="tradeall_live_{s}.png" alt="{s}">'
            f'<img class="state-overlay" src="tradeall_live_{s}_state.png" alt="{s} state">'
            f'</div></div>' for s in symbols)
        bt_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tradeall — backtest</title>
<style>
{_BASE_CHART_CSS}
</style></head>
<body>
<h2>tradeall — backtest ({os.path.basename(directory)})</h2>
{blocks}
<script>
setInterval(function() {{
  document.querySelectorAll('img').forEach(function(img) {{
    img.src = img.src.split('?')[0] + '?t=' + Date.now();
  }});
}}, 5000);
</script>
</body></html>"""
        html_path = os.path.join(directory, "tradeall_live_backtest.html")
        atomic_write_text(html_path, bt_html)
        mode = f"a sliding window of {window_hours}h" if window_hours is not None else "the whole interval"
        print(f"[tradeall_observe] BACKTEST ({mode}): {directory} | symbols: {symbols} | "
              f"rendering every {interval_sec}s")
        print(f"[tradeall_observe] open in browser: {html_path}")
        last_memory_trim = 0.0
        wh_val = window_hours or 0.0
        try:
            while True:
                for symbol in symbols:
                    try:
                        price_ts, price_vals = load_backtest_price_samples(directory, symbol)
                        trend_events = load_backtest_trend_starts(directory, symbol)
                        order_events = load_backtest_order_events(directory, symbol)
                        all_ts = price_ts + [e["ts"] for e in trend_events] + [e["ts"] for e in order_events]
                        if not all_ts:
                            window_start, window_end = time.time() - DAY_SECONDS, time.time()
                            label = "backtest (full replayed interval)"
                        elif wh_val > 0.0:
                            window_end = max(all_ts)
                            window_start = window_end - wh_val * 3600
                            try:
                                ts_str = datetime.fromtimestamp(window_end).strftime('%Y-%m-%d %H:%M')
                            except (ValueError, OSError, OverflowError):
                                ts_str = "unknown"
                            label = f"a sliding window of {wh_val}h (simulated clock: {ts_str})"
                        else:
                            window_start, window_end = min(all_ts), max(all_ts)
                            label = "backtest (full replayed interval)"
                        render_chart(
                            symbol, label, window_start, window_end,
                            price_ts, price_vals, trend_events, order_events,
                            os.path.join(directory, f"tradeall_live_{symbol}.png"),
                            shadow_events=_parse_shadow_rows(_read_pipe_log(os.path.join(directory, "tradeall_shadow.log"), 9), symbol),
                        )
                        state_text = build_backtest_state_text(directory, symbol)
                        if state_text:
                            render_state_image(state_text,
                                                os.path.join(directory, f"tradeall_live_{symbol}_state.png"))
                    except Exception as e:
                        print(f"[tradeall_observe] render error {symbol}: {e}")
                monotonic_now = time.monotonic()
                if monotonic_now - last_memory_trim >= MEMORY_TRIM_SECONDS:
                    _release_unused_memory()
                    last_memory_trim = monotonic_now
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n[tradeall_observe] stopped.")
        return

    active_summary = _get_active_summary()
    write_html(symbols, live_minutes=live_minutes, active_summary=active_summary)
    html_path = os.path.join(LIVE_OUT_DIR, "tradeall_live.html")
    print(f"[tradeall_observe] symbols: {symbols} | rendering every {interval_sec}s")
    print(f"[tradeall_observe] open in browser: {html_path}")

    # Schedule live charts and analysis state every cycle for a real-time feel.
    # Refresh day/week charts less often because their visible state barely changes
    # in three seconds and rendering them increased a cycle from ~3s to ~17s.
    last_day = last_week = last_memory_trim = 0.0
    try:
        while True:
            cycle_start = time.time()
            sample_current_prices(symbols)
            due_day = cycle_start - last_day >= day_refresh
            due_week = cycle_start - last_week >= week_refresh
            if due_day:
                active_summary = _get_active_summary()
                write_html(symbols, live_minutes=live_minutes, active_summary=active_summary)
            for symbol in symbols:
                try:
                    chart_specs = [
                        (
                            f"LIVE last {live_minutes:.0f} min",
                            live_minutes * 60,
                            os.path.join(LIVE_OUT_DIR, f"tradeall_live_{symbol}_live.png"),
                        )
                    ]
                    if due_day:
                        chart_specs.append(
                            (
                                "last 24h",
                                DAY_SECONDS,
                                os.path.join(LIVE_OUT_DIR, f"tradeall_live_{symbol}_day.png"),
                            )
                        )
                    if due_week:
                        chart_specs.append(
                            (
                                "last 7 days",
                                WEEK_SECONDS,
                                os.path.join(
                                    LIVE_OUT_DIR, f"tradeall_live_{symbol}_week.png"
                                ),
                            )
                        )
                    render_symbol_charts_live(symbol, chart_specs, window_end=cycle_start)
                    entry = _lookup_symbol_dict(read_json_dict(CACHE_TREND_PATH, default={}, recover_previous=True), symbol)
                    if isinstance(entry, dict) and entry:
                        try:
                            sh = _lookup_symbol_dict(read_json_dict(get_kalman_state_path(), default={}, recover_previous=True), symbol)
                            if isinstance(sh, dict) and sh:
                                entry = {**entry, **{k: v for k, v in sh.items() if k not in ("ts", "price")}}
                        except Exception:
                            pass
                        age = max(0.0, time.time() - safe_float(entry.get("ts"), 0.0))
                        pending_info = None
                        exposure_info = None
                        if isinstance(active_summary, dict):
                            notional_root = active_summary.get("pending_notional")
                            by_sym_pending = notional_root.get("by_symbol") if isinstance(notional_root, dict) else None
                            pending_map = _lookup_symbol_dict(by_sym_pending, symbol)
                            if isinstance(pending_map, dict):
                                parts = []
                                buy_val = safe_pos_float(pending_map.get("buy"), default=0.0)
                                sell_val = safe_pos_float(pending_map.get("sell"), default=0.0)
                                if buy_val > 0:
                                    parts.append(f"BUY ${buy_val:.2f}")
                                if sell_val > 0:
                                    parts.append(f"SELL ${sell_val:.2f}")
                                if parts:
                                    pending_info = ", ".join(parts)
                            exposures_root = active_summary.get("net_exposures")
                            by_sym_exp = exposures_root.get("by_symbol") if isinstance(exposures_root, dict) else None
                            exp_map = _lookup_symbol_dict(by_sym_exp, symbol)
                            if isinstance(exp_map, dict):
                                qty_val = safe_float(exp_map.get("qty"), 0.0)
                                if abs(qty_val) > 1e-9:
                                    raw_cost = exp_map.get("cost")
                                    cost_str = f" (${safe_float(raw_cost, 0.0):.2f})" if raw_cost is not None else ""
                                    exposure_info = f"{qty_val}{cost_str}"
                        state_text = format_state_text(
                            entry, f"ANALYSIS NOW ({age:.0f}s ago)",
                            pending=pending_info, exposure=exposure_info,
                        )
                        render_state_image(state_text,
                                            os.path.join(LIVE_OUT_DIR, f"tradeall_live_{symbol}_state.png"))
                except Exception as e:
                    print(f"[tradeall_observe] render error {symbol}: {e}")
            if due_day:
                last_day = cycle_start
            if due_week:
                last_week = cycle_start
            monotonic_now = time.monotonic()
            if monotonic_now - last_memory_trim >= MEMORY_TRIM_SECONDS:
                _release_unused_memory()
                last_memory_trim = monotonic_now
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[tradeall_observe] stopped.")


if __name__ == "__main__":
    main()
