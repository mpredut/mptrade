"""
backtest_ranges.py — parseaza rangurile de test scrise ca text SIMPLU,
DIRECTLY above a parameter in any config file (23 Jul,
offline/research/UNIFIED_BACKTEST_PLAN.md, a user decision: plain text, NOT YAML/JSON).

The format (one line, above the parameter, in ANY file — .env, a .conf of the
INI, monitortrades.conf):

    # BACKTEST: 5.0, 6.0, 7.0, 8.0, 9.0
    mt.gain = 7.0

Generic over formats: it knows nothing about .env vs INI vs monitortrades.conf —
it only looks for the comment line "# BACKTEST: ..." IMMEDIATELY above a line
"key = value" / "key=value" (whitespace around "=" is optional),
through a simple regex on the key. The comment must be on the very previous
line — any other line between them (another comment, a blank line)
cancels it, so that it is not wrongly attributed to a different parameter.

INI-style files (instruments.conf) have `[NAME]` sections that REUSE
the same keys (e.g. mt.gain appears both in [BINANCE_BTC] and in
[BINANCE_TAO]) — the key returned is prefixed with the current section
("BINANCE_BTC.mt.gain"), so that they cannot be confused or overwrite one another. Files
WITHOUT sections (.env, monitortrades.conf) return the key unchanged.

Why NOT a separate YAML/JSON (an explicit user decision): the test range lives
NEXT TO the real value, in the same file a human reads anyway —
not in a separate sidecar that can silently drift from the real value.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

_SECTION_RE = re.compile(r'^\s*\[([^\]]+)\]\s*(?:[#;].*)?$')
_BACKTEST_RE = re.compile(r'^\s*#\s*BACKTEST:\s*(.+?)\s*$')
_KEY_RE = re.compile(r'^\s*(?:export\s+)?([A-Za-z0-9_.]+)\s*=\s*(.+?)\s*$')


def scan_backtest_ranges(path: str) -> Dict[str, List[str]]:
    """Returns {key: [values_as_strings]} for every "# BACKTEST: ..." line
    found immediately above a key=value line, in `path`. Keys from
    files with [NAME] sections (instruments.conf) are prefixed with the section
    ("BINANCE_BTC.mt.gain"). [] if the file is missing or has no
    annotation of this kind."""
    out: Dict[str, List[str]] = {}
    if not path or isinstance(path, (bool, int, float)) or not os.path.isfile(path):
        return out
    section: Optional[str] = None
    pending: Optional[List[str]] = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            sm = _SECTION_RE.match(line)
            if sm:
                section = sm.group(1).strip() or None
                pending = None
                continue
            m = _BACKTEST_RE.match(line)
            if m:
                raw_vals = m.group(1).split("#", 1)[0].split(";", 1)[0]
                vals = [v.strip() for v in raw_vals.split(",") if v.strip()]
                pending = vals if vals else None
                continue
            m2 = _KEY_RE.match(line)
            if m2 and pending:
                key = m2.group(1)
                full_key = f"{section}.{key}" if section else key
                out[full_key] = pending
                pending = None
            else:
                # any other line (blank, another comment, another key without
                # an annotation) breaks the link -- the comment must be RIGHT
                # above, not "somewhere further up".
                pending = None
    return out


def scan_all(paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """{path: {key: [values]}} for a list of config files."""
    return {p: scan_backtest_ranges(p) for p in paths}
