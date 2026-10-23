"""Minimal protobuf wire-format helpers for Antigravity store files (SSOT, stdlib only).

Used by the systemd store hook (agy_store_fix.py) and the agy store modules, so it must not import
anything outside the Python standard library.
"""
from __future__ import annotations

from typing import Iterable, List, Set, Tuple

Field = Tuple[int, int, object]   # (field number, wire type, int | bytes)

SUMMARY_TITLE_FIELD = 1           # TrajectorySummary.title (fallback display text)
SUMMARY_ANNOTATIONS_FIELD = 15    # TrajectorySummary.annotations (sub-field 1 = user-visible title)
SUMMARY_SOURCE_FIELD = 20         # TrajectorySummary.source (17 = CORTEX_TRAJECTORY_SOURCE_CLI, hidden in Web UI)


def enc_varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def dec_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    val = shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, pos
        shift += 7
    raise ValueError("truncated varint")


def parse_fields(buf: bytes) -> List[Field]:
    pos, out = 0, []
    while pos < len(buf):
        key, pos = dec_varint(buf, pos)
        tag, wire = key >> 3, key & 7
        if wire == 0:
            v, pos = dec_varint(buf, pos)
        elif wire == 2:
            n, pos = dec_varint(buf, pos)
            v = buf[pos:pos + n]
            pos += n
        elif wire == 1:
            v = buf[pos:pos + 8]
            pos += 8
        elif wire == 5:
            v = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.append((tag, wire, v))
    return out


def enc_fields(fields: Iterable[Field]) -> bytes:
    out = bytearray()
    for tag, wire, v in fields:
        out += enc_varint((tag << 3) | wire)
        if wire == 0:
            out += enc_varint(int(v))
        elif wire == 2:
            out += enc_varint(len(v)) + bytes(v)
        else:
            out += bytes(v)
    return bytes(out)


def summary_ui_visible(raw: bytes, title: str = "") -> bytes:
    """Return a TrajectorySummary without the CLI source marker and, if given, with ``title`` as its UI title.

    Other annotation sub-fields (e.g. last_user_view_time) are preserved.
    """
    keep, annot_sub = [], []
    for tag, wire, v in parse_fields(raw):
        if tag == SUMMARY_SOURCE_FIELD:
            continue
        if tag == SUMMARY_ANNOTATIONS_FIELD and wire == 2:
            annot_sub = parse_fields(v)
            continue
        keep.append((tag, wire, v))
    if title:
        annot_sub = [(1, 2, title.encode())] + [x for x in annot_sub if x[0] != 1]
    if annot_sub:
        keep.append((SUMMARY_ANNOTATIONS_FIELD, 2, enc_fields(annot_sub)))
    return enc_fields(keep)


def entry_cid(entry: bytes) -> str:
    """Conversation id of a summaries-file map entry (sub-field 1), or '' if absent/unparsable."""
    try:
        fields = parse_fields(entry)
    except (ValueError, IndexError):
        return ""
    for tag, wire, v in fields:
        if tag == 1 and wire == 2:
            return v.decode("utf-8", "replace")
    return ""


def drop_entries(buf: bytes, cids: Set[str]) -> Tuple[bytes, int]:
    """Remove every top-level map entry (summaries and file mod times) keyed by one of ``cids``."""
    kept, removed = [], 0
    for tag, wire, v in parse_fields(buf):
        if wire == 2 and entry_cid(v) in cids:
            removed += 1
            continue
        kept.append((tag, wire, v))
    return enc_fields(kept), removed
