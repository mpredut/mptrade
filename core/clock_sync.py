"""core/clock_sync.py -- Canonical 3-tier polymorphic Open-Closed clock-sync core API.

Architecture (Open-Closed Principle & 3-tier polymorphism):
- Level 1 pure abstract contracts:
    ``ClockProbeBase`` (any clock source: OS NTP daemon or exchange UTC endpoint)
    ``ClockRepairStrategyBase`` (any non-disruptive clock repair mechanism)
- Level 2 domain abstract bases:
    ``OsClockProbeBase`` (local OS time-sync daemons)
    ``VenueClockProbeBase`` (remote exchange HTTP UTC endpoints using RTT midpoint)
- Level 3 concrete leaf classes + declarative registries:
    ``ChronyOsClockProbe``, ``TimesyncdOsClockProbe``, ``HttpVenueClockProbe``,
    ``TradingAdminClockRepairStrategy``, ``SystemctlClockRepairStrategy``
    registered in ``ClockProbeRegistry`` / ``ClockRepairRegistry`` and coordinated
    by ``ClockSyncService``.

Adding a new OS clock backend (e.g. ptp4l/ntpd/w32tm), a new venue clock source, or
a new repair strategy requires only subclassing and registering -- zero edits to core
evaluation or coordinator logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import math
import os
import re
import shutil
import subprocess
import threading
import time as _time
from typing import Any, Optional

from cachepaths import (
    _parse_bool_flag,
    exact_nonneg_int,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from core.config import (
    _eff_root_config_path,
    optional_env,
    parse_dotenv,
    required_float_env,
    required_int_env,
)
from core.http import keepalive_json_get_client

TRADING_ADMIN_BIN: str = "/usr/local/sbin/trading-admin"

_TIMESYNC_DURATION_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)(us|ms|s|min)\b")
_TIMESYNC_FIELD_RE = re.compile(r"^\s*(Server|Stratum|Leap|Offset|Delay|Jitter)\s*:\s*(.+?)\s*$")
_TIMESYNC_UNIT_MS: dict[str, float] = {
    "us": 0.001,
    "ms": 1.0,
    "s": 1000.0,
    "min": 60_000.0,
}


# ---------------------------------------------------------------------------
# Immutable Value Objects & Formulaic Policy (SSOT)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClockSyncPolicy:
    """Formulaically correlated OS & venue clock-sync thresholds derived from ``config.env``."""

    warn_ms: float
    step_ms: float
    venue_warn_ms: float
    max_sync_rtt_ms: float
    consecutive_checks: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClockSample:
    """Normalized observation from either an OS NTP daemon or a remote venue UTC endpoint."""

    source: str
    kind: str  # "os" | "venue"
    ok: bool = True
    synchronized: Optional[bool] = None
    offset_ms: Optional[float] = None
    rtt_ms: Optional[int] = None
    resolution_ms: int = 1
    stratum: int = 0
    server_ms: Optional[int] = None
    last_offset_ms: Optional[float] = None
    rms_offset_ms: Optional[float] = None
    root_delay_ms: Optional[float] = None
    root_dispersion_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    freq_ppm: Optional[float] = None
    leap_status: str = ""
    server: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "os":
            out: dict[str, Any] = {
                "service": self.source,
                "synchronized": bool(self.synchronized),
                "stratum": self.stratum,
                "server": self.server,
                "offset_ms": self.offset_ms,
                "leap_status": self.leap_status,
            }
            for k in ("last_offset_ms", "rms_offset_ms", "root_delay_ms", "root_dispersion_ms", "jitter_ms", "freq_ppm"):
                if (v := getattr(self, k)) is not None:
                    out[k] = v
            if self.error:
                out["error"] = self.error
            return out
        out = {"ok": self.ok, "resolution_ms": self.resolution_ms}
        if self.ok:
            out.update({
                "server_ms": self.server_ms,
                "offset_ms": self.offset_ms,
                "rtt_ms": self.rtt_ms,
            })
        elif self.error:
            out["error"] = self.error
        return out


@dataclass(frozen=True, slots=True)
class ClockRepairResult:
    """Outcome of executing a ``ClockRepairStrategyBase`` implementation."""

    ok: bool
    strategy: str
    message: str
    returncode: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClockHealthReport:
    """Pure deterministic evaluation combining OS NTP health and venue UTC clock offsets."""

    os_clock: dict[str, Any]
    venue_probes: dict[str, dict[str, Any]]
    healthy: bool
    needs_repair: bool
    needs_makestep: bool
    severity: str
    reasons: tuple[str, ...]
    warn_ms: float
    step_ms: float
    venue_warn_ms: float
    max_abs_os_offset_ms: Optional[float]
    max_abs_venue_offset_ms: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_clock": dict(self.os_clock),
            "venue_probes": {k: dict(v) for k, v in self.venue_probes.items()},
            "healthy": self.healthy,
            "needs_repair": self.needs_repair,
            "needs_makestep": self.needs_makestep,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "warn_ms": self.warn_ms,
            "step_ms": self.step_ms,
            "venue_warn_ms": self.venue_warn_ms,
            "max_abs_os_offset_ms": self.max_abs_os_offset_ms,
            "max_abs_venue_offset_ms": self.max_abs_venue_offset_ms,
        }


def resolve_clock_sync_policy() -> ClockSyncPolicy:
    """Load clock-sync anchor thresholds from root ``config.env`` and derive correlated bounds."""
    cfg_d = parse_dotenv(_eff_root_config_path())
    warn_raw = optional_env("CLOCK_OFFSET_WARN_MS") or cfg_d.get("CLOCK_OFFSET_WARN_MS")
    consec_raw = optional_env("CLOCK_SYNC_CONSECUTIVE") or cfg_d.get("CLOCK_SYNC_CONSECUTIVE")
    warn_ms = required_float_env("CLOCK_OFFSET_WARN_MS", {"CLOCK_OFFSET_WARN_MS": warn_raw})
    consec = required_int_env("CLOCK_SYNC_CONSECUTIVE", {"CLOCK_SYNC_CONSECUTIVE": consec_raw})
    if warn_ms <= 0.0:
        raise ValueError("CLOCK_OFFSET_WARN_MS must be positive")
    if consec <= 0:
        raise ValueError("CLOCK_SYNC_CONSECUTIVE must be positive")
    return ClockSyncPolicy(
        warn_ms=warn_ms,
        step_ms=round(warn_ms * 10.0, 3),
        venue_warn_ms=round(warn_ms * 4.0, 3),
        max_sync_rtt_ms=round(warn_ms * 20.0, 3),
        consecutive_checks=consec,
    )


# ---------------------------------------------------------------------------
# Pure Venue-Agnostic Measurement, Parsers & Deterministic Evaluator (SSOT)
# ---------------------------------------------------------------------------

def measure_rtt_compensated_offset_ms(
    fetch_server_ms_fn: Callable[[], Optional[int]],
    *,
    now_ms_fn: Optional[Callable[[], int]] = None,
    max_rtt_ms: Optional[float] = None,
    max_attempts: int = 2,
) -> tuple[int, int, int]:
    """Measure ``(server_ms, raw_offset_ms, rtt_ms)`` using RTT-midpoint compensation.

    If the first sample's RTT exceeds ``max_rtt_ms`` and ``max_attempts > 1``, retries
    and keeps the sample with the lowest round-trip time so transient network spikes
    do not bias client or telemetry clock alignment.
    """
    clock_ms = now_ms_fn if callable(now_ms_fn) else (lambda: math.trunc(_time.time() * 1000))
    eff_attempts = max(1, exact_nonneg_int(max_attempts, default=1) or 1)
    rtt_cap = safe_pos_float(max_rtt_ms, default=None)

    def _once() -> tuple[int, int, int]:
        t0_ms = clock_ms()
        server_ms = fetch_server_ms_fn()
        if server_ms is None or isinstance(server_ms, bool) or server_ms <= 0:
            raise ValueError("invalid serverTime response")
        t1_ms = clock_ms()
        rtt_ms = max(0, t1_ms - t0_ms)
        local_mid_ms = (t0_ms + t1_ms) // 2
        return server_ms, server_ms - local_mid_ms, rtt_ms

    best = _once()
    if rtt_cap is not None and best[2] > rtt_cap and eff_attempts > 1:
        for _ in range(eff_attempts - 1):
            try:
                cand = _once()
                if cand[2] < best[2]:
                    best = cand
                if best[2] <= rtt_cap:
                    break
            except Exception:
                break
    return best


def _parse_timesync_duration_ms(raw: object) -> Optional[float]:
    """Parse a systemd timesync duration string (e.g. ``-3.216ms``, ``+512us``, ``1min 2.5s``) into ms."""
    if not isinstance(raw, str):
        return None
    txt = raw.strip()
    if not txt:
        return None
    sign = -1.0 if txt.startswith("-") else 1.0
    body = txt[1:].strip() if txt[0] in "+-" else txt
    total_ms = 0.0
    matched = False
    for num_s, unit in _TIMESYNC_DURATION_TOKEN_RE.findall(body):
        val = safe_nonneg_float(num_s, default=None)
        mult = _TIMESYNC_UNIT_MS.get(unit)
        if val is not None and mult is not None:
            total_ms += val * mult
            matched = True
    return round(sign * total_ms, 3) if matched else None


def parse_chronyc_tracking_csv(raw_csv: object) -> Optional[ClockSample]:
    """Parse one CSV line from ``chronyc -c tracking`` into a ``ClockSample``."""
    if not isinstance(raw_csv, str):
        return None
    line = next((ln.strip() for ln in raw_csv.splitlines() if ln.strip()), "")
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 14:
        return None
    stratum = exact_nonneg_int(parts[2], default=0) or 0
    sys_off_s = safe_float(parts[4], default=None)
    last_off_s = safe_float(parts[5], default=None)
    rms_off_s = safe_nonneg_float(parts[6], default=None)
    freq_ppm = safe_float(parts[7], default=None)
    root_delay_s = safe_nonneg_float(parts[10], default=None)
    root_disp_s = safe_nonneg_float(parts[11], default=None)
    leap = parts[13]
    synced = stratum > 0 and leap.lower() != "not synchronised"
    return ClockSample(
        source="chrony",
        kind="os",
        ok=True,
        synchronized=synced,
        stratum=stratum,
        server=parts[1] or parts[0],
        offset_ms=round(sys_off_s * 1000.0, 3) if sys_off_s is not None else None,
        last_offset_ms=round(last_off_s * 1000.0, 3) if last_off_s is not None else None,
        rms_offset_ms=round(rms_off_s * 1000.0, 3) if rms_off_s is not None else None,
        root_delay_ms=round(root_delay_s * 1000.0, 3) if root_delay_s is not None else None,
        root_dispersion_ms=round(root_disp_s * 1000.0, 3) if root_disp_s is not None else None,
        freq_ppm=round(freq_ppm, 3) if freq_ppm is not None else None,
        leap_status=leap,
    )


def parse_timesync_status_text(
    raw_text: object, *, ntp_synchronized: Optional[bool] = None
) -> Optional[ClockSample]:
    """Parse ``timedatectl timesync-status`` output into a ``ClockSample``."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    fields: dict[str, str] = {}
    for ln in raw_text.splitlines():
        if (m := _TIMESYNC_FIELD_RE.match(ln)) is not None:
            fields[m.group(1)] = m.group(2)
    if not fields and ntp_synchronized is None:
        return None
    stratum = exact_nonneg_int(fields.get("Stratum"), default=0) or 0
    leap = fields.get("Leap", "")
    synced = (
        bool(ntp_synchronized)
        if ntp_synchronized is not None
        else (stratum > 0 and leap.lower() not in ("", "unsynchronized", "not synchronised"))
    )
    return ClockSample(
        source="systemd-timesyncd",
        kind="os",
        ok=True,
        synchronized=synced,
        stratum=stratum,
        server=fields.get("Server", ""),
        offset_ms=_parse_timesync_duration_ms(fields.get("Offset")),
        root_delay_ms=_parse_timesync_duration_ms(fields.get("Delay")),
        jitter_ms=_parse_timesync_duration_ms(fields.get("Jitter")),
        leap_status=leap or ("normal" if synced else "unsynchronized"),
    )


def _as_sample_dict(item: object) -> dict[str, Any]:
    if isinstance(item, ClockSample):
        return item.to_dict()
    return dict(item) if isinstance(item, Mapping) else {}


def evaluate_clock_sync_health(
    os_clock: ClockSample | Mapping[str, Any] | None,
    venue_probes: Mapping[str, ClockSample | Mapping[str, Any]] | None = None,
    *,
    policy: ClockSyncPolicy | Mapping[str, float | int] | None = None,
) -> ClockHealthReport:
    """Pure deterministic evaluator combining OS NTP telemetry and optional venue UTC probes."""
    if isinstance(policy, ClockSyncPolicy):
        warn_ms, step_ms, venue_warn_ms = policy.warn_ms, policy.step_ms, policy.venue_warn_ms
    elif isinstance(policy, Mapping):
        warn_ms = safe_pos_float(policy.get("warn_ms"), default=None)
        step_ms = safe_pos_float(policy.get("step_ms"), default=None)
        venue_warn_ms = safe_pos_float(policy.get("venue_warn_ms"), default=None)
        if warn_ms is None or step_ms is None or venue_warn_ms is None:
            raise ValueError("clock_sync policy requires positive warn_ms, step_ms, and venue_warn_ms")
    else:
        pol = resolve_clock_sync_policy()
        warn_ms, step_ms, venue_warn_ms = pol.warn_ms, pol.step_ms, pol.venue_warn_ms

    os_dict = _as_sample_dict(os_clock)
    venue_dicts: dict[str, dict[str, Any]] = {
        str(k): _as_sample_dict(v) for k, v in sorted((venue_probes or {}).items())
    }

    reasons: list[str] = []
    needs_makestep = False
    os_offset_abs: Optional[float] = None

    if not os_dict:
        reasons.append("os_clock_unavailable")
    else:
        synced = _parse_bool_flag(os_dict.get("synchronized"), False)
        if not synced:
            reasons.append(f"os_ntp_unsynchronized({os_dict.get('service', 'unknown')})")
        off = safe_float(os_dict.get("offset_ms"), default=None)
        if off is not None:
            os_offset_abs = abs(off)
            if os_offset_abs >= warn_ms:
                reasons.append(f"os_offset_ms={off:+.1f}(limit={warn_ms:.0f})")
            if os_offset_abs >= step_ms:
                needs_makestep = True

    max_venue_abs: Optional[float] = None
    for v_key, p_info in venue_dicts.items():
        if not _parse_bool_flag(p_info.get("ok"), False):
            continue
        v_off = safe_float(p_info.get("offset_ms"), default=None)
        if v_off is None:
            continue
        v_rtt = safe_nonneg_float(p_info.get("rtt_ms"), default=0.0) or 0.0
        v_res = safe_nonneg_float(p_info.get("resolution_ms"), default=1.0) or 1.0
        abs_v = abs(v_off)
        max_venue_abs = abs_v if max_venue_abs is None else max(max_venue_abs, abs_v)
        eff_tol = venue_warn_ms + v_res + (v_rtt / 2.0)
        if abs_v >= eff_tol:
            reasons.append(f"{v_key}_offset_ms={v_off:+.0f}(tol={eff_tol:.0f},rtt={v_rtt:.0f})")
            if abs_v >= step_ms + v_res + (v_rtt / 2.0):
                needs_makestep = True

    healthy = not reasons
    severity = "INFO" if healthy else ("CRITICAL" if needs_makestep else "HIGH")
    return ClockHealthReport(
        os_clock=os_dict,
        venue_probes=venue_dicts,
        healthy=healthy,
        needs_repair=not healthy,
        needs_makestep=needs_makestep,
        severity=severity,
        reasons=tuple(reasons),
        warn_ms=warn_ms,
        step_ms=step_ms,
        venue_warn_ms=venue_warn_ms,
        max_abs_os_offset_ms=round(os_offset_abs, 3) if os_offset_abs is not None else None,
        max_abs_venue_offset_ms=round(max_venue_abs, 3) if max_venue_abs is not None else None,
    )


# ---------------------------------------------------------------------------
# Level 1 Pure Abstract Contracts (Open-Closed Extension Boundaries)
# ---------------------------------------------------------------------------

class ClockProbeBase(ABC):
    """Level-1 pure abstract contract for any OS or remote venue clock source."""

    name: str = ""
    kind: str = ""
    priority: int = 100

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when this clock source can be queried in the current environment."""

    @abstractmethod
    def probe(self, *, timeout_sec: float = 4.0) -> Optional[ClockSample]:
        """Collect one normalized ``ClockSample`` from this source."""


class ClockRepairStrategyBase(ABC):
    """Level-1 pure abstract contract for any non-disruptive OS/application clock repair strategy."""

    name: str = ""
    priority: int = 100

    @abstractmethod
    def can_apply(self) -> bool:
        """Return True when this repair strategy is applicable on the current host."""

    @abstractmethod
    def repair(self, *, needs_makestep: bool = False) -> ClockRepairResult:
        """Execute a bounded, non-disruptive clock resync without restarting the trading fleet."""


# ---------------------------------------------------------------------------
# Level 2 Domain-Specific Abstract Bases
# ---------------------------------------------------------------------------

class OsClockProbeBase(ClockProbeBase, ABC):
    """Level-2 abstract base for local OS time-sync daemons (chrony, timesyncd, ntpd, ptp4l)."""

    kind: str = "os"

    @staticmethod
    def _run_cmd(cmd: list[str], *, timeout_sec: float = 3.0) -> Optional[subprocess.CompletedProcess[str]]:
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=safe_pos_float(timeout_sec, default=3.0) or 3.0,
                check=False,
            )
        except Exception:
            return None


class VenueClockProbeBase(ClockProbeBase, ABC):
    """Level-2 abstract base for public venue UTC endpoints using RTT-midpoint compensation."""

    kind: str = "venue"
    url: str = ""
    resolution_ms: int = 1

    def is_available(self) -> bool:
        return bool(self.name and self.url)

    @abstractmethod
    def extract_server_time_ms(self, payload: Any) -> Optional[int]:
        """Extract normalized integer UTC epoch milliseconds from the raw endpoint payload."""

    def probe(
        self,
        *,
        timeout_sec: float = 4.0,
        fetch_json_fn: Optional[Callable[..., Any]] = None,
        now_ms_fn: Optional[Callable[[], int]] = None,
        max_rtt_ms: Optional[float] = None,
    ) -> ClockSample:
        eff_timeout = safe_pos_float(timeout_sec, default=4.0) or 4.0
        try:
            if callable(fetch_json_fn):
                server_ms, offset_ms, rtt_ms = measure_rtt_compensated_offset_ms(
                    lambda: self.extract_server_time_ms(fetch_json_fn(self.url, timeout=eff_timeout)),
                    now_ms_fn=now_ms_fn,
                    max_rtt_ms=max_rtt_ms,
                    max_attempts=2 if max_rtt_ms is not None else 1,
                )
            else:
                with keepalive_json_get_client() as ka_fetch:
                    # Warm the TCP+TLS socket once so handshake asymmetry does not bias the midpoint offset.
                    ka_fetch(self.url, timeout=eff_timeout)
                    server_ms, offset_ms, rtt_ms = measure_rtt_compensated_offset_ms(
                        lambda: self.extract_server_time_ms(ka_fetch(self.url, timeout=eff_timeout)),
                        now_ms_fn=now_ms_fn,
                        max_rtt_ms=max_rtt_ms,
                        max_attempts=2 if max_rtt_ms is not None else 1,
                    )
            return ClockSample(
                source=self.name,
                kind="venue",
                ok=True,
                server_ms=server_ms,
                offset_ms=offset_ms,
                rtt_ms=rtt_ms,
                resolution_ms=self.resolution_ms,
            )
        except Exception as exc:
            return ClockSample(
                source=self.name,
                kind="venue",
                ok=False,
                resolution_ms=self.resolution_ms,
                error=str(exc)[:120],
            )


# ---------------------------------------------------------------------------
# Level 3 Concrete Leaf Implementations
# ---------------------------------------------------------------------------

class ChronyOsClockProbe(OsClockProbeBase):
    """Leaf OS clock probe backed by ``chronyc -c tracking``."""

    name: str = "chrony"
    priority: int = 10

    def is_available(self) -> bool:
        return bool(shutil.which("chronyc"))

    def probe(self, *, timeout_sec: float = 3.0) -> Optional[ClockSample]:
        res = self._run_cmd(["chronyc", "-c", "tracking"], timeout_sec=timeout_sec)
        if res is not None and res.returncode == 0 and res.stdout.strip():
            return parse_chronyc_tracking_csv(res.stdout)
        return None


class TimesyncdOsClockProbe(OsClockProbeBase):
    """Leaf OS clock probe backed by ``timedatectl show`` + ``timedatectl timesync-status``."""

    name: str = "systemd-timesyncd"
    priority: int = 20

    def is_available(self) -> bool:
        return bool(shutil.which("timedatectl"))

    def probe(self, *, timeout_sec: float = 3.0) -> Optional[ClockSample]:
        ntp_synced: Optional[bool] = None
        res_sync = self._run_cmd(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], timeout_sec=timeout_sec)
        if res_sync is not None and res_sync.returncode == 0:
            ntp_synced = res_sync.stdout.strip().lower() == "yes"
        res_ts = self._run_cmd(["timedatectl", "timesync-status"], timeout_sec=timeout_sec)
        if res_ts is not None and res_ts.returncode == 0 and res_ts.stdout.strip():
            sample = parse_timesync_status_text(res_ts.stdout, ntp_synchronized=ntp_synced)
            if sample is not None:
                return sample
        if ntp_synced is not None:
            return ClockSample(
                source="systemd-timesyncd",
                kind="os",
                ok=True,
                synchronized=ntp_synced,
                stratum=0,
                offset_ms=None,
            )
        return None


class HttpVenueClockProbe(VenueClockProbeBase):
    """Declarative leaf venue probe constructed from ``(name, url, extract_fn, resolution_ms)``."""

    def __init__(
        self,
        name: str,
        url: str,
        extract_fn: Callable[[Any], Optional[int]],
        *,
        resolution_ms: int = 1,
        priority: int = 50,
    ) -> None:
        self.name = str(name).strip().lower()
        self.url = str(url).strip()
        self._extract_fn = extract_fn
        self.resolution_ms = max(1, exact_nonneg_int(resolution_ms, default=1) or 1)
        self.priority = priority

    def extract_server_time_ms(self, payload: Any) -> Optional[int]:
        return self._extract_fn(payload)


class TradingAdminClockRepairStrategy(ClockRepairStrategyBase):
    """Leaf repair strategy delegating to the scoped root wrapper ``trading-admin sync-clock``."""

    name: str = "trading-admin"
    priority: int = 10

    def __init__(self, *, admin_bin: str = TRADING_ADMIN_BIN) -> None:
        self.admin_bin = admin_bin

    def can_apply(self) -> bool:
        return bool(self.admin_bin and os.path.exists(self.admin_bin))

    def repair(self, *, needs_makestep: bool = False) -> ClockRepairResult:
        _ = needs_makestep
        cmd = ["sudo", "-n", self.admin_bin, "sync-clock"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            lines = (res.stdout or res.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else f"rc={res.returncode}"
            return ClockRepairResult(
                ok=res.returncode == 0,
                strategy=self.name,
                message=f"trading-admin sync-clock: {detail}",
                returncode=res.returncode,
            )
        except Exception as exc:
            return ClockRepairResult(ok=False, strategy=self.name, message=f"trading-admin sync-clock failed: {exc}", returncode=1)


class SystemctlClockRepairStrategy(ClockRepairStrategyBase):
    """Fallback repair strategy restarting chrony (or systemd-timesyncd) via ``sudo -n systemctl``."""

    name: str = "systemctl-ntp"
    priority: int = 20

    def can_apply(self) -> bool:
        return bool(shutil.which("systemctl"))

    def repair(self, *, needs_makestep: bool = False) -> ClockRepairResult:
        _ = needs_makestep
        svc = "chrony.service" if shutil.which("chronyd") else "systemd-timesyncd.service"
        cmd = ["sudo", "-n", "/bin/systemctl", "restart", svc]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            lines = (res.stdout or res.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else f"rc={res.returncode}"
            return ClockRepairResult(
                ok=res.returncode == 0,
                strategy=self.name,
                message=f"systemctl restart {svc}: {detail}",
                returncode=res.returncode,
            )
        except Exception as exc:
            return ClockRepairResult(ok=False, strategy=self.name, message=f"systemctl restart {svc} failed: {exc}", returncode=1)


# ---------------------------------------------------------------------------
# Declarative Registries & Unified Facade (Closed to Modification, Open to Extension)
# ---------------------------------------------------------------------------

class ClockProbeRegistry:
    """Thread-safe declarative registry for OS and venue clock probes plus lazy venue loaders."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._os_probes: dict[str, OsClockProbeBase] = {}
        self._venue_probes: dict[str, VenueClockProbeBase] = {}
        self._venue_loaders: dict[str, Callable[[], Iterable[VenueClockProbeBase]]] = {}

    def register_os_probe(self, probe: OsClockProbeBase) -> OsClockProbeBase:
        if not isinstance(probe, OsClockProbeBase) or not probe.name:
            raise TypeError("probe must be an OsClockProbeBase with a non-empty name")
        with self._lock:
            self._os_probes[probe.name] = probe
        return probe

    def register_venue_probe(self, probe: VenueClockProbeBase) -> VenueClockProbeBase:
        if not isinstance(probe, VenueClockProbeBase) or not probe.name:
            raise TypeError("probe must be a VenueClockProbeBase with a non-empty name")
        with self._lock:
            self._venue_probes[probe.name] = probe
        return probe

    def register_venue_loader(
        self, name: str, loader: Callable[[], Iterable[VenueClockProbeBase]]
    ) -> None:
        if not name or not callable(loader):
            raise TypeError("loader must have a non-empty name and be callable")
        with self._lock:
            self._venue_loaders[name] = loader

    def list_os_probes(self) -> tuple[OsClockProbeBase, ...]:
        with self._lock:
            probes = list(self._os_probes.values())
        return tuple(sorted(probes, key=lambda p: (p.priority, p.name)))

    def list_venue_probes(self) -> tuple[VenueClockProbeBase, ...]:
        with self._lock:
            combined: dict[str, VenueClockProbeBase] = dict(self._venue_probes)
            loaders = list(self._venue_loaders.values())
        for loader in loaders:
            try:
                for probe in loader() or ():
                    if isinstance(probe, VenueClockProbeBase) and probe.name and probe.name not in combined:
                        combined[probe.name] = probe
            except Exception:
                continue
        return tuple(sorted(combined.values(), key=lambda p: (p.priority, p.name)))


class ClockRepairRegistry:
    """Thread-safe declarative registry for non-disruptive clock repair strategies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strategies: dict[str, ClockRepairStrategyBase] = {}

    def register_strategy(self, strategy: ClockRepairStrategyBase) -> ClockRepairStrategyBase:
        if not isinstance(strategy, ClockRepairStrategyBase) or not strategy.name:
            raise TypeError("strategy must be a ClockRepairStrategyBase with a non-empty name")
        with self._lock:
            self._strategies[strategy.name] = strategy
        return strategy

    def list_strategies(self) -> tuple[ClockRepairStrategyBase, ...]:
        with self._lock:
            strats = list(self._strategies.values())
        return tuple(sorted(strats, key=lambda s: (s.priority, s.name)))


class ClockSyncService:
    """Canonical venue-agnostic coordinator for OS + venue clock probing, evaluation, and repair."""

    def __init__(
        self,
        *,
        probe_registry: Optional[ClockProbeRegistry] = None,
        repair_registry: Optional[ClockRepairRegistry] = None,
    ) -> None:
        self.probe_registry = probe_registry or DEFAULT_CLOCK_PROBE_REGISTRY
        self.repair_registry = repair_registry or DEFAULT_CLOCK_REPAIR_REGISTRY

    def collect_os_clock(self, *, timeout_sec: float = 3.0) -> Optional[ClockSample]:
        for probe in self.probe_registry.list_os_probes():
            try:
                if not probe.is_available():
                    continue
                if (sample := probe.probe(timeout_sec=timeout_sec)) is not None:
                    return sample
            except Exception:
                continue
        return None

    def probe_venues(
        self,
        *,
        timeout_sec: float = 4.0,
        fetch_json_fn: Optional[Callable[..., Any]] = None,
        now_ms_fn: Optional[Callable[[], int]] = None,
        max_rtt_ms: Optional[float] = None,
    ) -> dict[str, ClockSample]:
        out: dict[str, ClockSample] = {}
        for probe in self.probe_registry.list_venue_probes():
            if not probe.is_available():
                continue
            out[probe.name] = probe.probe(
                timeout_sec=timeout_sec,
                fetch_json_fn=fetch_json_fn,
                now_ms_fn=now_ms_fn,
                max_rtt_ms=max_rtt_ms,
            )
        return out

    def repair(self, *, needs_makestep: bool = False) -> ClockRepairResult:
        failures: list[ClockRepairResult] = []
        for strat in self.repair_registry.list_strategies():
            try:
                if not strat.can_apply():
                    continue
                res = strat.repair(needs_makestep=needs_makestep)
                if res.ok:
                    return res
                failures.append(res)
            except Exception as exc:
                failures.append(ClockRepairResult(ok=False, strategy=strat.name, message=str(exc), returncode=1))
        if failures:
            last = failures[-1]
            summary = " | ".join(f"{f.strategy}: {f.message}" for f in failures)
            return ClockRepairResult(ok=False, strategy=last.strategy, message=summary, returncode=last.returncode or 1)
        return ClockRepairResult(ok=False, strategy="none", message="no applicable clock repair strategy", returncode=1)

    def check(
        self,
        *,
        include_venue_probes: bool = False,
        auto_repair: bool = False,
        probe_timeout_sec: float = 4.0,
        policy: ClockSyncPolicy | Mapping[str, float | int] | None = None,
    ) -> tuple[ClockHealthReport, Optional[ClockRepairResult]]:
        eff_policy = policy if policy is not None else resolve_clock_sync_policy()
        eff_rtt_cap = (
            eff_policy.max_sync_rtt_ms
            if isinstance(eff_policy, ClockSyncPolicy)
            else safe_pos_float(eff_policy.get("max_sync_rtt_ms"), default=None)
        )
        os_sample = self.collect_os_clock(timeout_sec=min(3.0, probe_timeout_sec))
        v_samples = (
            self.probe_venues(timeout_sec=probe_timeout_sec, max_rtt_ms=eff_rtt_cap)
            if _parse_bool_flag(include_venue_probes, False)
            else {}
        )
        report = evaluate_clock_sync_health(os_sample, v_samples or None, policy=eff_policy)
        repair_res: Optional[ClockRepairResult] = None
        if _parse_bool_flag(auto_repair, False) and report.needs_repair:
            repair_res = self.repair(needs_makestep=report.needs_makestep)
            os_sample = self.collect_os_clock(timeout_sec=min(3.0, probe_timeout_sec))
            v_samples = (
                self.probe_venues(timeout_sec=probe_timeout_sec, max_rtt_ms=eff_rtt_cap)
                if _parse_bool_flag(include_venue_probes, False)
                else {}
            )
            report = evaluate_clock_sync_health(os_sample, v_samples or None, policy=eff_policy)
        return report, repair_res


DEFAULT_CLOCK_PROBE_REGISTRY = ClockProbeRegistry()
DEFAULT_CLOCK_PROBE_REGISTRY.register_os_probe(ChronyOsClockProbe())
DEFAULT_CLOCK_PROBE_REGISTRY.register_os_probe(TimesyncdOsClockProbe())

DEFAULT_CLOCK_REPAIR_REGISTRY = ClockRepairRegistry()
DEFAULT_CLOCK_REPAIR_REGISTRY.register_strategy(TradingAdminClockRepairStrategy())
DEFAULT_CLOCK_REPAIR_REGISTRY.register_strategy(SystemctlClockRepairStrategy())

DEFAULT_CLOCK_SYNC_SERVICE = ClockSyncService(
    probe_registry=DEFAULT_CLOCK_PROBE_REGISTRY,
    repair_registry=DEFAULT_CLOCK_REPAIR_REGISTRY,
)


def register_os_clock_probe(probe: OsClockProbeBase) -> OsClockProbeBase:
    """Register an additional OS clock probe without modifying core clock-sync code."""
    return DEFAULT_CLOCK_PROBE_REGISTRY.register_os_probe(probe)


def register_venue_clock_probe(probe: VenueClockProbeBase) -> VenueClockProbeBase:
    """Register an additional venue UTC clock probe without modifying core clock-sync code."""
    return DEFAULT_CLOCK_PROBE_REGISTRY.register_venue_probe(probe)


def register_venue_clock_loader(
    name: str, loader: Callable[[], Iterable[VenueClockProbeBase]]
) -> None:
    """Register a lazy discovery loader yielding ``VenueClockProbeBase`` instances."""
    DEFAULT_CLOCK_PROBE_REGISTRY.register_venue_loader(name, loader)


def register_clock_repair_strategy(strategy: ClockRepairStrategyBase) -> ClockRepairStrategyBase:
    """Register an additional non-disruptive clock repair strategy."""
    return DEFAULT_CLOCK_REPAIR_REGISTRY.register_strategy(strategy)
