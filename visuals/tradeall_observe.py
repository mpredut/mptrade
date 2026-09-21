#!/usr/bin/env python3
"""Separate observational monitor for the tradeall trend model.

This module neither imports nor starts tradeall.py or bapi_placeorder.py. It only
reads pipe-delimited logger files and cache_instant_trend.json.

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
import os
import tempfile
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGGER_DIR = os.path.join(ROOT, "logger")
CACHE_TREND_PATH = os.path.join(ROOT, "cachedb", "cache_instant_trend.json")

# Dedicated live-mode output directory (PNGs and HTML), deliberately separated
# from the repository. Unlike ROOT, it contains only generated charts and never
# .env files, keys, or cachedb, making it the only safe directory to serve.
LIVE_OUT_DIR = os.path.join(ROOT, "tradeall_live")

PRICE_SAMPLES_PREFIX = "tradeall_price_samples_"
DECISIONS_PREFIX = "tradeall_decisions_"
OUTCOMES_PREFIX = "order_outcomes_"
SHADOW_PREFIX = "tradeall_shadow_"
SHADOW_COLOR = "#8250df"   # violet — semnalele shadow (Kalman), distinct de verde/rosu

STATE_COLORS = {"UP": "#1a7f37", "DOWN": "#cf222e", "HOLD": "#8c8c8c"}
DAY_SECONDS = 24 * 3600
WEEK_SECONDS = 7 * DAY_SECONDS
MEMORY_TRIM_SECONDS = 300


def _sanitize_field(value):
    return str(value).replace("|", "/").replace("\n", " ")


def _daily_log_path(prefix, date):
    return os.path.join(LOGGER_DIR, f"{prefix}{date.isoformat()}.log")


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
            return cached[2]                   # neschimbat -> zero I/O

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
            break                               # linie incompleta (scriere in curs) — o reluam data viitoare
        consumed += len(raw)
        parts = raw.decode("utf-8", errors="replace").rstrip("\n").split("|")
        if len(parts) != ncols:
            continue
        rows.append(parts)
    _PIPE_LOG_CACHE[path] = (
        consumed,
        ncols,
        rows,
        read_stat.st_dev,
        read_stat.st_ino,
        read_stat.st_size,
        read_stat.st_mtime_ns,
    )
    _PIPE_LOG_CACHE.move_to_end(path)
    while len(_PIPE_LOG_CACHE) > _PIPE_LOG_CACHE_MAX_FILES:
        _PIPE_LOG_CACHE.popitem(last=False)
    return rows


def _log_dates(days_back):
    today = datetime.now().date()
    return [today - timedelta(days=i) for i in range(days_back)]


# -- Step B: independent price sampling that does not touch tradeall. ----------

_LAST_SAMPLED_TS = {}


def sample_current_prices(symbols):
    if not os.path.exists(CACHE_TREND_PATH):
        return
    try:
        with open(CACHE_TREND_PATH, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[tradeall_observe] eroare citire cache_instant_trend.json: {e}")
        return

    pending = []
    for symbol in symbols:
        entry = snapshot.get(symbol)
        if not entry or "current_price" not in entry:
            continue
        try:
            sample_ts = float(entry.get("ts", time.time()))
        except (TypeError, ValueError):
            continue
        # The loop runs every two seconds, but the source may publish less often.
        # Avoid recording the same tick repeatedly and inflating the log and RAM cache.
        if sample_ts <= _LAST_SAMPLED_TS.get(symbol, float("-inf")):
            continue
        pending.append((symbol, sample_ts, entry["current_price"]))
    if not pending:
        return

    os.makedirs(LOGGER_DIR, exist_ok=True)
    path = _daily_log_path(PRICE_SAMPLES_PREFIX, datetime.now().date())
    try:
        with open(path, "a", encoding="utf-8") as f:
            for symbol, sample_ts, current_price in pending:
                # Use the snapshot timestamp, not our clock. If the source freezes,
                # a current timestamp on stale data would create a false jagged block.
                f.write(f"{sample_ts}|{_sanitize_field(symbol)}|{current_price}\n")
                _LAST_SAMPLED_TS[symbol] = sample_ts
    except OSError as e:
        print(f"[tradeall_observe] error writing the price samples: {e}")


# -- Load logs (steps A / A2). -------------------------------------------------

def load_price_samples(symbol, days_back):
    ts_list, px_list = [], []
    for d in reversed(_log_dates(days_back)):
        for ts, sym_, price in _read_pipe_log(_daily_log_path(PRICE_SAMPLES_PREFIX, d), 3):
            if sym_ != symbol:
                continue
            try:
                ts_list.append(float(ts))
                px_list.append(float(price))
            except ValueError:
                continue
    return ts_list, px_list


def _load_cachedb_price_entries(filename, symbol, max_bytes=4 * 1024 * 1024):
    """Read Cache24 ``[[ts_ms, price], ...]`` entries from cachedb.

    Incremental long-term JSONL files may grow to hundreds of megabytes, so the
    weekly caller reads only their ``max_bytes`` tail. The legacy JSON format
    used by the live 24-hour cache remains supported unchanged.
    """
    path = os.path.join(ROOT, "cachedb", filename)
    if not os.path.exists(path):
        return []
    if filename.endswith(".jsonl"):
        entries = []
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                data = f.read().decode("utf-8", errors="replace")
            lines = data.split("\n")[1:] if size > max_bytes else data.split("\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("s") == symbol:
                        entries.append(rec["i"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        except OSError:
            return []
        return entries
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("items", {}).get(symbol, [])
    except (OSError, json.JSONDecodeError, ValueError):
        return []   # Another process may be writing the file; skip this cycle.


def _load_history_jsonl_tail(symbol, max_bytes=4 * 1024 * 1024):
    """Read enough sparse long-term history for a seven-day window.

    The approximately 40 MB file covers about eleven months. Its final
    ``max_bytes`` span weeks, avoiding a full scan on every cycle.
    """
    path = os.path.join(ROOT, "cachedb", f"cache_price_{symbol}.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        # Drop the first line only when reading began in the middle of the file.
        # Otherwise this would discard the first valid record in a small file.
        lines = data.split("\n")[1:] if size > max_bytes else data.split("\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("s") == symbol:
                    entries.append(rec["i"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    except OSError:
        pass
    return entries


def load_price_series_live(symbol, days_back, include_history=True):
    """Merge and sort all existing sources into an immediately populated chart.

    Sources comprise sparse long-term history, the dense long-term archive, this
    monitor's samples, and the fleet's dense live 24-hour cache. Dense points win
    timestamp collisions. ``include_history=False`` skips sparse history for
    windows already covered by dense caches, saving parsing on each cycle.
    """
    # Merge into one-second buckets because sources have roughly one-second clock
    # offsets. Without alignment, slightly different prices form a false jagged
    # block. Dense sources are written last and therefore win each bucket.
    points = {}
    # Add sparse/archive sources first, then dense live data, so an old archive
    # record cannot overwrite the current tick from the same second.
    if include_history:
        for entry in _load_history_jsonl_tail(symbol):
            try:
                points[int(entry[0] / 1000.0)] = float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
    # The unbounded long-term archive grows continuously. Parsing it every
    # two-second live cycle was wasteful because the live file already provides
    # identical density for 24 hours. Load it only for longer windows where it
    # contributes data; this path previously accounted for 46% sustained CPU.
    if include_history:
        for entry in _load_cachedb_price_entries(f"cache_24price_long_{symbol}.jsonl", symbol):
            try:
                points[int(entry[0] / 1000.0)] = float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
    own_ts, own_px = load_price_samples(symbol, days_back)
    for t, p in zip(own_ts, own_px):
        points[int(t)] = p
    for entry in _load_cachedb_price_entries(f"cache_24price_{symbol}.json", symbol):
        try:
            points[int(entry[0] / 1000.0)] = float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
    ts_sorted = sorted(points)
    return ts_sorted, [points[t] for t in ts_sorted]


def load_trend_starts(symbol, days_back):
    events = []
    for d in reversed(_log_dates(days_back)):
        for row in _read_pipe_log(_daily_log_path(DECISIONS_PREFIX, d), 7):
            ts, sym_, event, state, old_state, price, prev_confirm = row
            if sym_ != symbol or event != "trend_start":
                continue
            try:
                events.append({"ts": float(ts), "state": state, "old_state": old_state,
                                "price": float(price), "prev_confirm_count": prev_confirm})
            except ValueError:
                continue
    events.sort(key=lambda e: e["ts"])
    return events


def load_order_events(symbol, days_back):
    """Load only tradeall.py events from the fleet-wide outcome log."""
    events = []
    for d in reversed(_log_dates(days_back)):
        for row in _read_pipe_log(_daily_log_path(OUTCOMES_PREFIX, d), 9):
            ts, sym_, side, price, qty, outcome, refuse_reason, caller, motivation = row
            if sym_ != symbol or caller != "tradeall.py":
                continue
            try:
                events.append({
                    "ts": float(ts), "side": side, "price": float(price),
                    "outcome": outcome, "reason": motivation or refuse_reason,
                })
            except ValueError:
                continue
    events.sort(key=lambda e: e["ts"])
    return events


def _parse_shadow_rows(rows, symbol):
    """Parse shadow rows: ts|symbol|signal|event|state|old_state|price|vel|vel_std."""
    events = []
    for row in rows:
        ts, sym_, signal, event, state, old_state, price, vel, vel_std = row
        if sym_ != symbol or event != "trend_start":
            continue
        try:
            events.append({"ts": float(ts), "signal": signal, "state": int(state),
                            "old_state": int(old_state), "price": float(price),
                            "vel": vel, "vel_std": vel_std})
        except ValueError:
            continue
    events.sort(key=lambda e: e["ts"])
    return events


def load_shadow_events(symbol, days_back):
    rows = []
    for d in reversed(_log_dates(days_back)):
        rows.extend(_read_pipe_log(_daily_log_path(SHADOW_PREFIX, d), 9))
    return _parse_shadow_rows(rows, symbol)


def load_backtest_shadow_events(directory, symbol):
    return _parse_shadow_rows(_read_pipe_log(os.path.join(directory, "tradeall_shadow.log"), 9), symbol)


def build_trend_regions(events, window_start, window_end):
    """Build state regions from each trend start to the next (plan steps A/C)."""
    regions = []
    for i, ev in enumerate(events):
        end = events[i + 1]["ts"] if i + 1 < len(events) else window_end
        start, end = max(ev["ts"], window_start), min(end, window_end)
        if end > start:
            regions.append((start, end, ev["state"]))
    return regions


# -- Load flat, non-rotating backtest logs. ------------------------------------

def load_backtest_price_samples(directory, symbol):
    ts_list, px_list = [], []
    for ts, sym_, price in _read_pipe_log(os.path.join(directory, "tradeall_price_samples.log"), 3):
        if sym_ != symbol:
            continue
        try:
            ts_list.append(float(ts))
            px_list.append(float(price))
        except ValueError:
            continue
    return ts_list, px_list


def load_backtest_trend_starts(directory, symbol):
    events = []
    for row in _read_pipe_log(os.path.join(directory, "tradeall_decisions.log"), 7):
        ts, sym_, event, state, old_state, price, prev_confirm = row
        if sym_ != symbol or event != "trend_start":
            continue
        try:
            events.append({"ts": float(ts), "state": state, "old_state": old_state,
                            "price": float(price), "prev_confirm_count": prev_confirm})
        except ValueError:
            continue
    events.sort(key=lambda e: e["ts"])
    return events


def load_backtest_order_events(directory, symbol):
    """Load one backtest run; its sole caller needs no caller filtering."""
    events = []
    for row in _read_pipe_log(os.path.join(directory, "order_outcomes.log"), 9):
        ts, sym_, side, price, qty, outcome, refuse_reason, caller, motivation = row
        if sym_ != symbol:
            continue
        try:
            events.append({
                "ts": float(ts), "side": side, "price": float(price),
                "outcome": outcome, "reason": motivation or refuse_reason,
            })
        except ValueError:
            continue
    events.sort(key=lambda e: e["ts"])
    return events


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
    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tradeall_observe_", suffix=".png", dir=directory)
    os.close(fd)
    try:
        fig.savefig(temp_path, format="png", **savefig_kwargs)
        os.replace(temp_path, out_path)
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


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
        f"BUY  acceptat: {sum(1 for e in accepted if e['side'] == 'BUY')}   "
        f"refused: {sum(1 for e in refused if e['side'] == 'BUY')}\n"
        f"SELL acceptat: {sum(1 for e in accepted if e['side'] == 'SELL')}   "
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

    ax.set_xlim(datetime.fromtimestamp(window_start), datetime.fromtimestamp(window_end))
    ax.set_title(f"{symbol} — {window_label}  (actualizat {datetime.now().strftime('%H:%M:%S')})")
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


def format_state_text(entry, header):
    trend_map = {1: "UP", -1: "DOWN", 0: "FLAT"}
    text = (
        f"{header}\n"
        f"price:     {entry.get('current_price', '?')}\n"
        f"trend:     {trend_map.get(entry.get('final_trend'), '?')}\n"
        f"grad rec:  {entry.get('gradient_recent', 0):+.4f}\n"
        f"slope mic: {entry.get('slope_small', 0):+.3f}\n"
        f"slope mare:{entry.get('slope_big', 0):+.3f}\n"
        f"epsilon:   {entry.get('epsilon', 0):.4f}"
    )
    # Add shadow rows only when their keys exist in the snapshot.
    if entry.get("kalman_trend") is not None:
        text += (f"\nkalman:    {trend_map.get(entry.get('kalman_trend'), '?')} "
                 f"(v={entry.get('kalman_vel', 0):+.3f}%/min ±{entry.get('kalman_vel_std', 0):.3f})")
    v1h = entry.get("vol_1h_pct")
    if v1h is not None:
        text += (f"\nvol 1h:    {v1h:.2f}% → re:{entry.get('adapt_reentry_pct', '?')}% "
                 f"dca:{entry.get('adapt_dca_pct', '?')}%")
    return text


def build_analysis_state_text(symbol):
    """Combine current live tradeall analysis with its shadow state.

    tradeall writes shadow_state.json directly because cacheManager owns the
    trend cache file.
    """
    try:
        with open(CACHE_TREND_PATH, "r", encoding="utf-8") as f:
            entry = json.load(f).get(symbol)
    except (OSError, json.JSONDecodeError):
        entry = None
    if not entry:
        return None
    try:
        with open(os.path.join(ROOT, "cachedb", "shadow_state.json"), "r", encoding="utf-8") as f:
            sh = json.load(f).get(symbol)
        if sh:
            entry = {**entry, **{k: v for k, v in sh.items() if k not in ("ts", "price")}}
    except (OSError, json.JSONDecodeError):
        pass
    age = time.time() - entry.get("ts", 0)
    return format_state_text(entry, f"ANALYSIS NOW ({age:.0f}s ago)")


def build_backtest_state_text(directory, symbol):
    """Load simulated analysis state written by offline/backtests/tradeall.py."""
    path = os.path.join(directory, "analysis_state.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f).get(symbol)
    except (OSError, json.JSONDecodeError):
        entry = None
    if not entry:
        return None
    sim_time = datetime.fromtimestamp(entry.get("ts", 0)).strftime("%m-%d %H:%M:%S")
    return format_state_text(entry, f"ANALIZA SIMULATA ({sim_time})")


@_close_new_figures
def render_state_image(state_text, out_path):
    """Render the state box separately for the HTML chart's hover display."""
    fig = plt.figure(figsize=(4.6, 2.5))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.96, state_text, va="top", ha="left", fontsize=11, family="monospace",
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
    window_end = time.time() if window_end is None else window_end
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


def render_symbol_chart_live(symbol, window_label, window_seconds, out_path):
    """Provide compatibility for callers requesting a single window."""
    render_symbol_charts_live(symbol, [(window_label, window_seconds, out_path)])


def render_symbol_chart_backtest(symbol, directory, out_path, window_hours=None):
    """Render either all replayed data or a sliding backtest window.

    With ``window_hours``, anchor the window at the most recent simulated
    timestamp. Repeated calls while the backtester runs follow its simulated
    clock, making events appear exactly when replay reaches them.
    """
    price_ts, price_vals = load_backtest_price_samples(directory, symbol)
    trend_events = load_backtest_trend_starts(directory, symbol)
    order_events = load_backtest_order_events(directory, symbol)

    all_ts = price_ts + [e["ts"] for e in trend_events] + [e["ts"] for e in order_events]
    if not all_ts:
        window_start, window_end = time.time() - DAY_SECONDS, time.time()
        label = "backtest (tot intervalul reluat)"
    elif window_hours:
        window_end = max(all_ts)
        window_start = window_end - window_hours * 3600
        label = (f"a sliding window of {window_hours}h (simulated clock: "
                 f"{datetime.fromtimestamp(window_end).strftime('%Y-%m-%d %H:%M')})")
    else:
        window_start, window_end = min(all_ts), max(all_ts)
        label = "backtest (tot intervalul reluat)"

    render_chart(symbol, label, window_start, window_end,
                 price_ts, price_vals, trend_events, order_events, out_path,
                 shadow_events=load_backtest_shadow_events(directory, symbol))


def render_backtest_chunks(symbol, directory, chunk_hours=24, out_dir=None):
    """Generate one detailed frame per ``chunk_hours`` instead of one dense chart.

    Each frame contains fewer events, preventing label overlap while covering the
    complete replayed interval.
    """
    out_dir = out_dir or directory
    price_ts, price_vals = load_backtest_price_samples(directory, symbol)
    trend_events = load_backtest_trend_starts(directory, symbol)
    order_events = load_backtest_order_events(directory, symbol)

    all_ts = price_ts + [e["ts"] for e in trend_events] + [e["ts"] for e in order_events]
    if not all_ts:
        print(f"[tradeall_observe] {symbol}: no data for the frames yet")
        return []

    chunk_sec = chunk_hours * 3600
    range_start, range_end = min(all_ts), max(all_ts)
    n_chunks = int((range_end - range_start) // chunk_sec) + 1

    paths = []
    for i in range(n_chunks):
        c_start = range_start + i * chunk_sec
        c_end = min(c_start + chunk_sec, range_end)
        label = f"{datetime.fromtimestamp(c_start).strftime('%Y-%m-%d %H:%M')} " \
                f"→ {datetime.fromtimestamp(c_end).strftime('%Y-%m-%d %H:%M')}  (cadru {i+1}/{n_chunks})"
        fname = f"tradeall_live_{symbol}_frame{i+1:03d}_{datetime.fromtimestamp(c_start).strftime('%Y%m%d_%H%M')}.png"
        path = os.path.join(out_dir, fname)
        render_chart(symbol, label, c_start, c_end, price_ts, price_vals,
                     trend_events, order_events, path)
        paths.append(path)
    return paths


# -- Static, serverless HTML with day/week toggle and auto-refresh. ------------

def write_html(symbols, live_minutes=60):
    blocks = []
    for s in symbols:
        blocks.append(f'''
  <div class="chart">
    <h3>{s}</h3>
    <div class="chart-wrap">
      <img class="view-live" src="tradeall_live_{s}_live.png" alt="{s} live">
      <img class="view-day" src="tradeall_live_{s}_ziua.png" alt="{s} zi" style="display:none">
      <img class="view-week" src="tradeall_live_{s}_saptamana.png" alt="{s} saptamana" style="display:none">
      <img class="state-overlay" src="tradeall_live_{s}_state.png" alt="{s} stare">
    </div>
  </div>''')

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tradeall — live</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#111; color:#eee; margin: 16px; }}
img {{ max-width: 100%; border: 1px solid #333; border-radius: 4px; }}
button {{ margin: 4px 6px 14px 0; padding: 6px 16px; border-radius: 4px; border: 1px solid #444;
         background:#222; color:#eee; cursor:pointer; }}
button.active {{ background:#1f6feb; border-color:#1f6feb; }}
.chart {{ margin-bottom: 24px; }}
.chart-wrap {{ position: relative; display: inline-block; max-width: 100%; }}
.state-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                  opacity: 0; transition: opacity .15s; pointer-events: none;
                  max-width: 45%; border: none; }}
.chart-wrap:hover .state-overlay {{ opacity: 1; }}
</style></head>
<body>
<h2>tradeall — monitor live</h2>
<button id="btn-live" class="active" onclick="show('live')">Live ({live_minutes:.0f} min)</button>
<button id="btn-day" onclick="show('day')">Zi</button>
<button id="btn-week" onclick="show('week')">Saptamana</button>
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
    os.makedirs(LIVE_OUT_DIR, exist_ok=True)
    with open(os.path.join(LIVE_OUT_DIR, "tradeall_live.html"), "w", encoding="utf-8") as f:
        f.write(html)


def write_backtest_html(directory, symbols):
    """Write minimal backtest HTML without a day/week toggle."""
    os.makedirs(directory, exist_ok=True)   # The backtester may not have created it yet.
    blocks = "".join(
        f'<div class="chart"><h3>{s}</h3><div class="chart-wrap">'
        f'<img src="tradeall_live_{s}.png" alt="{s}">'
        f'<img class="state-overlay" src="tradeall_live_{s}_state.png" alt="{s} stare">'
        f'</div></div>' for s in symbols)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>tradeall — backtest</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#111; color:#eee; margin: 16px; }}
img {{ max-width: 100%; border: 1px solid #333; border-radius: 4px; }}
.chart {{ margin-bottom: 24px; }}
.chart-wrap {{ position: relative; display: inline-block; max-width: 100%; }}
.state-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                  opacity: 0; transition: opacity .15s; pointer-events: none;
                  max-width: 45%; border: none; }}
.chart-wrap:hover .state-overlay {{ opacity: 1; }}
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
    with open(os.path.join(directory, "tradeall_live_backtest.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    from instrument_registry import symbols_for

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
                              "genereaza o SERIE de cadre STATICE (imagini), cate unul per N ore. "
                              "Generate once and exit (it does not loop).")
    parser.add_argument("--window-hours", type=float, default=None,
                         help="with --backtest-dir (without --frame-hours): a SLIDING window of N hours, "
                              "anchored to the current simulated clock — it runs in a loop (like live), "
                              "sliding as offline/backtests/tradeall.py writes new data in parallel; "
                              "the events appear on the frame exactly when the backtester reaches them)")
    args = parser.parse_args()
    symbols = (symbols_for("binance") if args.symbols is None else
               [s.strip() for s in args.symbols.split(",") if s.strip()])

    if args.backtest_dir and args.frame_hours:
        directory = os.path.abspath(args.backtest_dir)
        for symbol in symbols:
            paths = render_backtest_chunks(symbol, directory, chunk_hours=args.frame_hours)
            print(f"[tradeall_observe] {symbol}: {len(paths)} cadre generate in {directory}")
            for p in paths:
                print(f"    {p}")
        return

    if args.backtest_dir:
        directory = os.path.abspath(args.backtest_dir)
        write_backtest_html(directory, symbols)
        html_path = os.path.join(directory, "tradeall_live_backtest.html")
        mode = f"a sliding window of {args.window_hours}h" if args.window_hours else "the whole interval"
        print(f"[tradeall_observe] BACKTEST ({mode}): {directory} | simboluri: {symbols} | "
              f"randare la {args.interval}s")
        print(f"[tradeall_observe] deschide in browser: {html_path}")
        last_memory_trim = 0.0
        try:
            while True:
                for symbol in symbols:
                    try:
                        render_symbol_chart_backtest(
                            symbol, directory, os.path.join(directory, f"tradeall_live_{symbol}.png"),
                            window_hours=args.window_hours)
                        state_text = build_backtest_state_text(directory, symbol)
                        if state_text:
                            render_state_image(state_text,
                                                os.path.join(directory, f"tradeall_live_{symbol}_state.png"))
                    except Exception as e:
                        print(f"[tradeall_observe] eroare randare {symbol}: {e}")
                monotonic_now = time.monotonic()
                if monotonic_now - last_memory_trim >= MEMORY_TRIM_SECONDS:
                    _release_unused_memory()
                    last_memory_trim = monotonic_now
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[tradeall_observe] stopped.")
        return

    write_html(symbols, live_minutes=args.live_minutes)
    html_path = os.path.join(LIVE_OUT_DIR, "tradeall_live.html")
    print(f"[tradeall_observe] simboluri: {symbols} | randare la {args.interval}s")
    print(f"[tradeall_observe] deschide in browser: {html_path}")

    # Schedule live charts and analysis state every cycle for a real-time feel.
    # Refresh day/week charts less often because their visible state barely changes
    # in three seconds and rendering them increased a cycle from ~3s to ~17s.
    last_day = last_week = last_memory_trim = 0.0
    try:
        while True:
            cycle_start = time.time()
            sample_current_prices(symbols)
            due_day = cycle_start - last_day >= args.day_refresh
            due_week = cycle_start - last_week >= args.week_refresh
            for symbol in symbols:
                try:
                    chart_specs = [
                        (
                            f"LIVE ultimele {args.live_minutes:.0f} min",
                            args.live_minutes * 60,
                            os.path.join(LIVE_OUT_DIR, f"tradeall_live_{symbol}_live.png"),
                        )
                    ]
                    if due_day:
                        chart_specs.append(
                            (
                                "ultimele 24h",
                                DAY_SECONDS,
                                os.path.join(LIVE_OUT_DIR, f"tradeall_live_{symbol}_ziua.png"),
                            )
                        )
                    if due_week:
                        chart_specs.append(
                            (
                                "ultimele 7 zile",
                                WEEK_SECONDS,
                                os.path.join(
                                    LIVE_OUT_DIR, f"tradeall_live_{symbol}_saptamana.png"
                                ),
                            )
                        )
                    render_symbol_charts_live(symbol, chart_specs, window_end=cycle_start)
                    state_text = build_analysis_state_text(symbol)
                    if state_text:
                        render_state_image(state_text,
                                            os.path.join(LIVE_OUT_DIR, f"tradeall_live_{symbol}_state.png"))
                except Exception as e:
                    print(f"[tradeall_observe] eroare randare {symbol}: {e}")
            if due_day:
                last_day = cycle_start
            if due_week:
                last_week = cycle_start
            monotonic_now = time.monotonic()
            if monotonic_now - last_memory_trim >= MEMORY_TRIM_SECONDS:
                _release_unused_memory()
                last_memory_trim = monotonic_now
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[tradeall_observe] stopped.")


if __name__ == "__main__":
    main()
