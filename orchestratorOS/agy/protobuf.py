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
SUMMARY_STATUS_FIELD = 28         # TrajectorySummary.status (2 = RUNNING, 4 = IDLE)


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
        if shift >= 64:
            raise ValueError("varint too long")
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
        if tag <= 0:
            raise ValueError("invalid field tag 0")
        if wire == 0:
            v, pos = dec_varint(buf, pos)
        elif wire == 2:
            n, pos = dec_varint(buf, pos)
            if pos + n > len(buf):
                raise ValueError("truncated length-delimited field")
            v = buf[pos:pos + n]
            pos += n
        elif wire == 1:
            if pos + 8 > len(buf):
                raise ValueError("truncated fixed64 field")
            v = buf[pos:pos + 8]
            pos += 8
        elif wire == 5:
            if pos + 4 > len(buf):
                raise ValueError("truncated fixed32 field")
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


def summary_ui_visible(raw: bytes, title: str = "", *, reset_running: bool = False) -> bytes:
    """Return a TrajectorySummary without the CLI source marker and, if given, with ``title`` as its UI title.

    Other annotation sub-fields (e.g. last_user_view_time) are preserved. When ``reset_running`` is True,
    a stale RUNNING status (field 28 = 2) is reset to IDLE (4) and fields are sorted by tag.
    """
    keep, annot_sub = [], []
    try:
        top_fields = parse_fields(raw)
    except (ValueError, IndexError):
        top_fields = []
    for tag, wire, v in top_fields:
        if tag == SUMMARY_SOURCE_FIELD:
            continue
        if tag == SUMMARY_ANNOTATIONS_FIELD and wire == 2:
            try:
                annot_sub = parse_fields(v)
            except (ValueError, IndexError):
                annot_sub = []
            continue
        if reset_running and tag == SUMMARY_STATUS_FIELD and wire == 0 and v == 2:
            v = 4
        keep.append((tag, wire, v))
    if title:
        annot_sub = [(1, 2, title.encode())] + [x for x in annot_sub if x[0] != 1]
    if annot_sub:
        keep.append((SUMMARY_ANNOTATIONS_FIELD, 2, enc_fields(annot_sub)))
    if reset_running:
        keep.sort(key=lambda x: x[0])
    return enc_fields(keep)


def summary_annotation_title(raw: bytes) -> str:
    """Extract the user-visible annotation title (field 15, sub-field 1) from a TrajectorySummary."""
    try:
        top_fields = parse_fields(raw)
    except (ValueError, IndexError):
        return ""
    for tag, wire, v in top_fields:
        if tag == SUMMARY_ANNOTATIONS_FIELD and wire == 2:
            try:
                sub_fields = parse_fields(v)
            except (ValueError, IndexError):
                continue
            for t, w, x in sub_fields:
                if t == 1 and w == 2:
                    return x.decode("utf-8", "ignore").strip()
    return ""


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


def list_entry_cids(buf: bytes) -> Set[str]:
    """Return all conversation IDs stored in top-level summary map entries (field 1) of a summaries .pb file."""
    try:
        top_fields = parse_fields(buf)
    except (ValueError, IndexError):
        return set()
    return {
        cid
        for tag, wire, v in top_fields
        if tag == 1 and wire == 2 and (cid := entry_cid(v))
    }


def drop_entries(buf: bytes, cids: Set[str]) -> Tuple[bytes, int]:
    """Remove every top-level map entry (summaries and file mod times) keyed by one of ``cids``."""
    try:
        top_fields = parse_fields(buf)
    except (ValueError, IndexError):
        return bytes(buf or b""), 0
    kept, removed = [], 0
    for tag, wire, v in top_fields:
        if wire == 2 and entry_cid(v) in cids:
            removed += 1
            continue
        kept.append((tag, wire, v))
    return enc_fields(kept), removed


def replace_summary_entry(buf: bytes, cid: str, raw_summary: bytes) -> Tuple[bytes, bool]:
    """Update the stored TrajectorySummary of ``cid`` in place if present; never adds new conversation IDs."""
    try:
        top_fields = parse_fields(buf)
    except (ValueError, IndexError):
        return bytes(buf or b""), False
    out, updated = [], False
    for tag, wire, v in top_fields:
        if tag == 1 and wire == 2 and entry_cid(v) == cid:
            out.append((1, 2, enc_fields([(1, 2, cid.encode()), (2, 2, raw_summary)])))
            updated = True
        else:
            out.append((tag, wire, v))
    return (enc_fields(out), True) if updated else (buf, False)

