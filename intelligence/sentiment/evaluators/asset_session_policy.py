"""3-tier polymorphic AssetSessionPolicy hierarchy (Crypto, TokenizedEquity, RthEquity).

Level 1: ``AssetClassPolicyInterface`` (pure abstract contract).
Level 2: ``AssetSessionPolicy`` and ``Continuous24x7SessionPolicy`` (domain abstract bases).
Level 3: ``Crypto24x7SessionPolicy``, ``TokenizedEquity24x7SessionPolicy``,
         ``EquityRthSessionPolicy`` registered in ``AssetClassPolicyRegistry``.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from botcore import (
    _prices_diverged,
    resolve_us_equity_session_state,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from instrument_registry import resolve_symbol_identity
from intelligence.macro_sentiment_ssot import (
    SHORT_KINEMATIC_METRIC_KEYS,
    parse_ts_px_row,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    STALE_SHORT_24H_KEYS,
    _SHORT_SLOPE_OVERRIDE_KEYS,
    compute_24h_price_metrics,
    normalize_and_dedup_24h_series,
    override_series_tail_close,
    pct_delta,
    prepend_older_non_diverged_points,
    prune_closed_equity_stale_metrics,
    restore_rth_metric_anchors,
    sort_ts_rows,
    strip_off_rth_flat_poller_tail,
)


@dataclass(frozen=True)
class AssetTelemetryCapabilities:
    """Single Source of Truth (SSOT) declaring telemetry & execution capabilities per asset class.

    Centralizes L2 orderbook availability, perpetual/liquidation support, synthetic spot CVD
    gating, CBOE options OI handling, and Section 3B book labels so evaluators, TCA, and prompt
    renderers never scatter ad-hoc ``is_equity`` / ``oi_usd`` / ``spot_cvd_source`` checks.
    """

    has_real_l2_book: bool
    has_crypto_perpetuals: bool
    has_liquidation_clusters: bool
    has_real_spot_tape: bool
    book_source_prefix: str
    bid_zone_label: str
    ask_zone_label: str
    suppress_frozen_ob_guard: bool
    allow_synthetic_spot_cvd: bool
    uses_cboe_options_oi: bool
    has_zero_funding: bool

    def has_live_options_oi(self, minfo: Dict[str, Any]) -> bool:
        """Return whether ``minfo`` carries positive live CBOE options open interest."""
        return self.uses_cboe_options_oi and (safe_float(minfo.get("oi_usd"), 0.0) or 0.0) > 0.0

    def allow_spot_cvd_display(self, minfo: Dict[str, Any], *, has_perp_cvd: bool) -> bool:
        """Return whether ``spot_cvd_15m`` should be rendered for this asset class and source."""
        is_real_spot = minfo.get("spot_cvd_source") == "real_spot"
        if not self.allow_synthetic_spot_cvd and not is_real_spot:
            return False
        return bool(has_perp_cvd or is_real_spot)

    def allow_whale_long_display(self, minfo: Dict[str, Any]) -> bool:
        """Return whether ``whale_long_pct`` represents real positioning (crypto perp or CBOE OI)."""
        return self.has_crypto_perpetuals or self.has_live_options_oi(minfo)

    def allow_whale_long_delta_display(self, minfo: Dict[str, Any], *, has_flow: bool) -> bool:
        """Return whether ``d_whale_long`` in ``cycle_delta`` represents real positioning movement."""
        if self.has_crypto_perpetuals:
            return True
        if self.uses_cboe_options_oi:
            if "oi_usd" in minfo:
                return (safe_float(minfo.get("oi_usd"), 0.0) or 0.0) > 0.0
            return bool(has_flow)
        return False

    def resolve_deriv_display_mode(self, minfo: Dict[str, Any]) -> str:
        """Return canonical Section 3B derivatives display mode: ``crypto_perp``, ``cboe_options``, or ``none``."""
        if self.has_crypto_perpetuals:
            return "crypto_perp"
        if self.has_live_options_oi(minfo):
            return "cboe_options"
        return "none"


_CRYPTO_TELEMETRY_CAPS = AssetTelemetryCapabilities(
    has_real_l2_book=True,
    has_crypto_perpetuals=True,
    has_liquidation_clusters=True,
    has_real_spot_tape=True,
    book_source_prefix="",
    bid_zone_label="bid_wall",
    ask_zone_label="ask_wall",
    suppress_frozen_ob_guard=False,
    allow_synthetic_spot_cvd=True,
    uses_cboe_options_oi=False,
    has_zero_funding=False,
)

_TOKENIZED_EQUITY_TELEMETRY_CAPS = AssetTelemetryCapabilities(
    has_real_l2_book=True,
    has_crypto_perpetuals=False,
    has_liquidation_clusters=False,
    has_real_spot_tape=True,
    book_source_prefix="",
    bid_zone_label="bid_wall",
    ask_zone_label="ask_wall",
    suppress_frozen_ob_guard=False,
    allow_synthetic_spot_cvd=True,
    uses_cboe_options_oi=False,
    has_zero_funding=True,
)

_RTH_EQUITY_TELEMETRY_CAPS = AssetTelemetryCapabilities(
    has_real_l2_book=False,
    has_crypto_perpetuals=False,
    has_liquidation_clusters=False,
    has_real_spot_tape=False,
    book_source_prefix="ob_proxy[yahoo_5m_vol_profile] ",
    bid_zone_label="est_bid_zone",
    ask_zone_label="est_ask_zone",
    suppress_frozen_ob_guard=True,
    allow_synthetic_spot_cvd=False,
    uses_cboe_options_oi=True,
    has_zero_funding=True,
)


class AssetClassPolicyInterface(ABC):
    """Pure abstract contract (Level 1) for asset-class market hours, derivatives availability, and series normalization."""

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Canonical asset-class key ('crypto', 'tokenized_equity', 'rth_equity')."""

    @property
    @abstractmethod
    def telemetry_capabilities(self) -> AssetTelemetryCapabilities:
        """Declarative telemetry & L2 execution capability contract for this asset class."""

    @property
    @abstractmethod
    def is_equity(self) -> bool:
        """Whether this asset class follows cash-equity session rules (RTH / extended hours)."""

    @property
    @abstractmethod
    def is_tokenized_equity(self) -> bool:
        """Whether this asset is a 24/7 spot tokenized equity (xStock)."""

    @property
    @abstractmethod
    def supports_extended_hours(self) -> bool:
        """Whether this asset class has distinct pre-market / after-hours sessions outside regular hours."""

    @abstractmethod
    def resolve_session_state(self, now_ts: Optional[float] = None) -> str:
        """Return canonical session state ('24X7', 'PRE', 'REGULAR', 'POST', or 'CLOSED') for this asset class."""

    @property
    @abstractmethod
    def has_crypto_perpetuals(self) -> bool:
        """Whether crypto perpetual futures & whale L/S positioning exist for this asset class."""

    @property
    @abstractmethod
    def has_liquidation_clusters(self) -> bool:
        """Whether perpetual liquidation cluster heatmaps apply to this asset class."""

    @property
    @abstractmethod
    def uses_equity_risk_priors(self) -> bool:
        """Whether portfolio risk models use equity beta/correlation/VIX priors instead of crypto beta priors."""

    @property
    @abstractmethod
    def allow_live_vp_ohlc_network(self) -> bool:
        """Whether on-demand pretrade VP OHLC backfill may hit crypto spot exchange REST endpoints."""

    @property
    @abstractmethod
    def ohlc_vp_live_clock_skew_ms(self) -> float:
        """Maximum series-vs-wallclock gap (ms) considered live for TPO->REAL_VOL OHLC upgrades."""

    @property
    @abstractmethod
    def price_divergence_threshold(self) -> float:
        """Relative price divergence threshold above which cached series are rejected as split/corrupt."""

    @abstractmethod
    def clean_raw_24h_series(self, rows: list, symbol: str, *, is_market_open_fn: Any = None) -> list:
        """Normalize raw 24h series according to asset-class session rules."""

    @abstractmethod
    def merge_ohlc24_into_raw_series(self, raw_ser: list, ohlc24_pts: list) -> list:
        """Merge stitched 24h OHLC bars into the raw 24h series."""

    @abstractmethod
    def evaluate_post_rth_override(
        self,
        matched_series_24h: list,
        m24: Dict[str, Any],
        sym_px_cur: float,
        now_ts: float,
        *,
        entry: Optional[Dict[str, Any]] = None,
        compute_24h_fn: Any = None,
    ) -> Tuple[list, Dict[str, Any], Optional[float], bool]:
        """Evaluate post-session price override while preserving session Volume Profile anchors."""

    @abstractmethod
    def prune_stale_short_metrics(
        self,
        entry: Optional[Dict[str, Any]],
        m24: Dict[str, Any],
        matched_series_24h: list,
        now_ts: float,
        *,
        post_rth_series_replaced: bool = False,
        rth_fresh_override: Optional[bool] = None,
        include_kinematics: bool = False,
        has_series_tail: bool = True,
    ) -> None:
        """Prune stale short-horizon slopes when session bars are no longer live."""


class AssetSessionPolicy(AssetClassPolicyInterface):
    """Domain-specific abstract base (Level 2) for asset-class session rules and series normalization."""

    cls_name = "AssetSessionPolicy"

    def __init__(self) -> None:
        if type(self) in (AssetSessionPolicy, Continuous24x7SessionPolicy):
            raise TypeError(f"Cannot instantiate abstract class {type(self).__name__}")

    @property
    def is_tokenized_equity(self) -> bool:
        return False

    @property
    def has_crypto_perpetuals(self) -> bool:
        return self.telemetry_capabilities.has_crypto_perpetuals

    @property
    def has_liquidation_clusters(self) -> bool:
        return self.telemetry_capabilities.has_liquidation_clusters

    @property
    def uses_equity_risk_priors(self) -> bool:
        return True

    def normalize_24h_series(self, series_24h: list, symbol: str, *, is_market_open_fn: Any = None) -> list:
        """Deduplicate and session-normalize the 24h series."""
        return normalize_and_dedup_24h_series(series_24h, symbol, is_market_open_fn=is_market_open_fn)

    def apply_post_rth_price_override(
        self,
        matched_series_24h: list,
        m24: Dict[str, Any],
        cur_px: float,
        *,
        entry: Optional[Dict[str, Any]] = None,
        compute_24h_fn: Any = None,
        now_ts: Optional[float] = None,
    ) -> Tuple[list, Dict[str, Any], bool]:
        """Apply an off-RTH quote override while preserving session Volume Profile anchors."""
        return matched_series_24h, m24, False


class Continuous24x7SessionPolicy(AssetSessionPolicy):
    """Domain-specific abstract base (Level 2) for 24/7 continuous spot venues (crypto & tokenized equities)."""

    cls_name = "Continuous24x7SessionPolicy"

    @property
    def is_equity(self) -> bool:
        return False

    @property
    def supports_extended_hours(self) -> bool:
        return False

    def resolve_session_state(self, now_ts: Optional[float] = None) -> str:
        _ = now_ts
        return "24X7"

    @property
    def allow_live_vp_ohlc_network(self) -> bool:
        return True

    @property
    def ohlc_vp_live_clock_skew_ms(self) -> float:
        return 30 * 60 * 1000.0

    @property
    def price_divergence_threshold(self) -> float:
        return 0.10

    def clean_raw_24h_series(self, rows: list, symbol: str, *, is_market_open_fn: Any = None) -> list:
        return sort_ts_rows(rows)

    def merge_ohlc24_into_raw_series(self, raw_ser: list, ohlc24_pts: list) -> list:
        if not raw_ser or not ohlc24_pts:
            return raw_ser
        ordered_raw = sort_ts_rows(raw_ser)
        cur_p0 = parse_ts_px_row(ordered_raw[0])
        cur_p1 = parse_ts_px_row(ordered_raw[-1])
        if cur_p0 is None or cur_p1 is None:
            return raw_ser
        ohlc_2col: list[list[float]] = []
        for row in sort_ts_rows(ohlc24_pts):
            if (pt := parse_ts_px_row(row)) is not None:
                ohlc_2col.append([pt[0], pt[1]])
        if len(ohlc_2col) < 2:
            return raw_ser
        ohlc_last = ohlc_2col[-1]
        if (
            (cur_p1[0] - ohlc_last[0]) > 2 * 3600 * 1000
            or ohlc_last[0] > cur_p1[0] + 15 * 60 * 1000
            or _prices_diverged(ohlc_last[1], cur_p1[1], threshold=self.price_divergence_threshold)
        ):
            return raw_ser
        win_start_ms = cur_p1[0] - 86_400_000.0
        older_pts = [pt for pt in ohlc_2col if win_start_ms <= pt[0] < cur_p0[0]]
        if not older_pts:
            return raw_ser
        if _prices_diverged(older_pts[-1][1], cur_p0[1], threshold=self.price_divergence_threshold):
            return raw_ser
        return older_pts + ordered_raw

    def evaluate_post_rth_override(
        self,
        matched_series_24h: list,
        m24: Dict[str, Any],
        sym_px_cur: float,
        now_ts: float,
        *,
        entry: Optional[Dict[str, Any]] = None,
        compute_24h_fn: Any = None,
    ) -> Tuple[list, Dict[str, Any], Optional[float], bool]:
        return matched_series_24h, m24, None, False

    def prune_stale_short_metrics(
        self,
        entry: Optional[Dict[str, Any]],
        m24: Dict[str, Any],
        matched_series_24h: list,
        now_ts: float,
        *,
        post_rth_series_replaced: bool = False,
        rth_fresh_override: Optional[bool] = None,
        include_kinematics: bool = False,
        has_series_tail: bool = True,
    ) -> None:
        return None


class Crypto24x7SessionPolicy(Continuous24x7SessionPolicy):
    """Concrete policy (Level 3) for 24/7 continuous crypto spot & perpetual instruments."""

    cls_name = "Crypto24x7SessionPolicy"

    @property
    def asset_class(self) -> str:
        return "crypto"

    @property
    def telemetry_capabilities(self) -> AssetTelemetryCapabilities:
        return _CRYPTO_TELEMETRY_CAPS

    @property
    def uses_equity_risk_priors(self) -> bool:
        return False


class TokenizedEquity24x7SessionPolicy(Continuous24x7SessionPolicy):
    """Concrete policy (Level 3) for 24/7 spot tokenized equities (Kraken xStocks) lacking crypto perpetual markets."""

    cls_name = "TokenizedEquity24x7SessionPolicy"

    @property
    def asset_class(self) -> str:
        return "tokenized_equity"

    @property
    def telemetry_capabilities(self) -> AssetTelemetryCapabilities:
        return _TOKENIZED_EQUITY_TELEMETRY_CAPS

    @property
    def is_tokenized_equity(self) -> bool:
        return True


class EquityRthSessionPolicy(AssetSessionPolicy):
    """Concrete policy (Level 3) for RTH-gated cash equities (`_EQ` / `_US_EQ`)."""

    cls_name = "EquityRthSessionPolicy"

    @property
    def asset_class(self) -> str:
        return "rth_equity"

    @property
    def telemetry_capabilities(self) -> AssetTelemetryCapabilities:
        return _RTH_EQUITY_TELEMETRY_CAPS

    @property
    def is_equity(self) -> bool:
        return True

    @property
    def supports_extended_hours(self) -> bool:
        return True

    def resolve_session_state(self, now_ts: Optional[float] = None) -> str:
        return resolve_us_equity_session_state(now_ts)

    @property
    def allow_live_vp_ohlc_network(self) -> bool:
        return False

    @property
    def ohlc_vp_live_clock_skew_ms(self) -> float:
        return 72 * 3600 * 1000.0

    @property
    def price_divergence_threshold(self) -> float:
        return 0.35

    def clean_raw_24h_series(self, rows: list, symbol: str, *, is_market_open_fn: Any = None) -> list:
        ordered = sort_ts_rows(rows)
        if len(ordered) >= 5:
            pos_vol_indices = [
                i
                for i, r in enumerate(ordered)
                if isinstance(r, (list, tuple)) and len(r) >= 3 and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
            ]
            if len(pos_vol_indices) >= 4 and (first_pos_idx := pos_vol_indices[0]) > 0:
                first_real_pt = parse_ts_px_row(ordered[first_pos_idx])
                if first_real_pt is not None and first_real_pt[1] > 0.0:
                    lead_pts = [pt for r in ordered[:first_pos_idx] if (pt := parse_ts_px_row(r)) is not None]
                    last_lead_pt = lead_pts[-1] if lead_pts else None
                    has_continuous_ext_session = bool(
                        last_lead_pt is not None
                        and len({round(pt[1], 4) for pt in lead_pts}) >= 4
                        and (first_real_pt[0] - last_lead_pt[0]) <= 30 * 60 * 1000
                        and abs(last_lead_pt[1] - first_real_pt[1]) / first_real_pt[1] <= 0.05
                    )
                    if not has_continuous_ext_session:
                        trim_start = 0
                        for idx in range(first_pos_idx):
                            lead_pt = parse_ts_px_row(ordered[idx])
                            if lead_pt is None:
                                trim_start = idx + 1
                                continue
                            gap_ms = first_real_pt[0] - lead_pt[0]
                            px_rel_diff = abs(lead_pt[1] - first_real_pt[1]) / first_real_pt[1]
                            if gap_ms > 30 * 60 * 1000 or px_rel_diff > 0.05:
                                trim_start = idx + 1
                            else:
                                break
                        if trim_start > 0:
                            ordered = ordered[trim_start:]
        return strip_off_rth_flat_poller_tail(ordered, symbol, is_market_open_fn=is_market_open_fn)

    def merge_ohlc24_into_raw_series(self, raw_ser: list, ohlc24_pts: list) -> list:
        ohlc_bars = [
            r
            for r in sort_ts_rows(ohlc24_pts or [])
            if isinstance(r, (list, tuple)) and len(r) >= 3 and safe_nonneg_float(r[2], default=None) is not None
        ]
        if (
            sum(1 for r in ohlc_bars if (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0) >= 4
            and (ohlc_0 := parse_ts_px_row(ohlc_bars[0])) is not None
            and (ohlc_last := parse_ts_px_row(ohlc_bars[-1])) is not None
            and (ser_last := parse_ts_px_row(raw_ser[-1])) is not None
            and not _prices_diverged(ohlc_last[1], ser_last[1])
        ):
            by_bar_ts = {}
            for r_s in raw_ser:
                if (pt_s := parse_ts_px_row(r_s)) is not None:
                    if (
                        (isinstance(r_s, (list, tuple)) and len(r_s) >= 3 and safe_nonneg_float(r_s[2], default=None) is not None)
                        or pt_s[0] < ohlc_0[0]
                        or pt_s[0] > ohlc_last[0]
                        or (pt_s[0] >= ohlc_last[0] - 15 * 60 * 1000 and not _prices_diverged(pt_s[1], ohlc_last[1]))
                    ):
                        by_bar_ts[pt_s[0]] = r_s
            for r_o in ohlc_bars:
                if (pt_o := parse_ts_px_row(r_o)) is not None:
                    by_bar_ts[pt_o[0]] = r_o
            return [by_bar_ts[k] for k in sorted(by_bar_ts)]
        return raw_ser

    def apply_post_rth_price_override(
        self,
        matched_series_24h: list,
        m24: Dict[str, Any],
        cur_px: float,
        *,
        entry: Optional[Dict[str, Any]] = None,
        compute_24h_fn: Any = None,
        now_ts: Optional[float] = None,
    ) -> Tuple[list, Dict[str, Any], bool]:
        eff_compute_24h = compute_24h_fn or compute_24h_price_metrics
        s24_p0 = parse_ts_px_row(matched_series_24h[0]) if len(matched_series_24h) >= 2 else None
        s24_p1 = parse_ts_px_row(matched_series_24h[-1]) if len(matched_series_24h) >= 2 else None
        replaced = False
        override_px = round(cur_px, 4)
        if s24_p1 is not None and len(matched_series_24h) >= 2:
            prev_m24 = dict(m24)
            override_series_24h = override_series_tail_close(matched_series_24h, override_px)
            m24 = (
                eff_compute_24h(override_series_24h, is_equity=True, now_ts=now_ts)
                if eff_compute_24h is compute_24h_price_metrics
                else eff_compute_24h(override_series_24h)
            ) or m24
            restore_rth_metric_anchors(
                m24,
                vp_src=prev_m24,
                vwap_src=prev_m24,
                atr_src=prev_m24,
                live_px=override_px,
            )
            if abs(cur_px - s24_p1[1]) / max(cur_px, s24_p1[1]) > 0.005:
                replaced = True
                matched_series_24h = override_series_24h
                if isinstance(entry, dict):
                    for k_ov in _SHORT_SLOPE_OVERRIDE_KEYS:
                        entry.pop(k_ov, None)
        if "chg_24h_pct" in m24 and s24_p0 is not None and s24_p0[1] > 0:
            m24["chg_24h_pct"] = pct_delta(override_px, s24_p0[1])
        for k_stale in STALE_SHORT_24H_KEYS:
            m24.pop(k_stale, None)
        return matched_series_24h, m24, replaced

    @staticmethod
    def _is_series_tail_stale_ms(bar_ts_ms: int, now_ts: float, *, max_age_ms: int = 20 * 60 * 1000) -> bool:
        now_ms = math.trunc(now_ts * 1000)
        return bar_ts_ms > (now_ms + 60_000) or (now_ms - bar_ts_ms) > max_age_ms

    def evaluate_post_rth_override(
        self,
        matched_series_24h: list,
        m24: Dict[str, Any],
        sym_px_cur: float,
        now_ts: float,
        *,
        entry: Optional[Dict[str, Any]] = None,
        compute_24h_fn: Any = None,
    ) -> Tuple[list, Dict[str, Any], Optional[float], bool]:
        s24_p1 = parse_ts_px_row(matched_series_24h[-1]) if len(matched_series_24h) >= 2 else None
        post_rth_px_override: Optional[float] = None
        post_rth_series_replaced = False
        s24_tail_stale = bool(s24_p1 is not None and self._is_series_tail_stale_ms(s24_p1[0], now_ts))
        if (
            m24
            and s24_p1 is not None
            and (
                s24_tail_stale
                or (sym_px_cur > 0 and abs(sym_px_cur - s24_p1[1]) / max(sym_px_cur, s24_p1[1]) > 0.005)
            )
        ):
            if (
                sym_px_cur > 0
                and not _prices_diverged(sym_px_cur, s24_p1[1], threshold=self.price_divergence_threshold)
                and abs(round(sym_px_cur, 4) - round(s24_p1[1], 4)) > 1e-4
            ):
                post_rth_px_override = round(sym_px_cur, 4)
                matched_series_24h, m24, post_rth_series_replaced = self.apply_post_rth_price_override(
                    matched_series_24h, m24, sym_px_cur, entry=entry, compute_24h_fn=compute_24h_fn, now_ts=now_ts
                )
            else:
                for k_stale in STALE_SHORT_24H_KEYS:
                    m24.pop(k_stale, None)
                if s24_tail_stale:
                    m24.pop("chg_15m_pct", None)
        return matched_series_24h, m24, post_rth_px_override, post_rth_series_replaced

    def prune_stale_short_metrics(
        self,
        entry: Optional[Dict[str, Any]],
        m24: Dict[str, Any],
        matched_series_24h: list,
        now_ts: float,
        *,
        post_rth_series_replaced: bool = False,
        rth_fresh_override: Optional[bool] = None,
        include_kinematics: bool = False,
        has_series_tail: bool = True,
    ) -> None:
        if not isinstance(entry, dict):
            return
        prune_closed_equity_stale_metrics(m24, entry, quote_age_min=entry.get("quote_age_min"))
        if post_rth_series_replaced:
            return
        if not any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct")):
            for k_stale_90 in (
                "slope_90s",
                "growth_90s",
                "trend_90s",
                *(("vel_90s_pct_min", "accel_90s_pct_min2") if include_kinematics else ()),
            ):
                entry.pop(k_stale_90, None)
        if not has_series_tail:
            return
        is_series_stale = (
            (not rth_fresh_override)
            if rth_fresh_override is not None
            else bool(
                matched_series_24h
                and (last_eq_pt := parse_ts_px_row(matched_series_24h[-1])) is not None
                and self._is_series_tail_stale_ms(last_eq_pt[0], now_ts)
            )
        )
        if not any(k in m24 for k in ("chg_1m_pct", "chg_3m_pct", "chg_5m_pct")) or is_series_stale:
            for k_stale_s in (
                "slope_90s",
                "growth_90s",
                "trend_90s",
                "slope_3_7m",
                "growth_3_7m",
                "trend_3_7m",
                "gradient_recent_fast",
                *(SHORT_KINEMATIC_METRIC_KEYS if include_kinematics else ()),
            ):
                entry.pop(k_stale_s, None)


class AssetClassPolicyRegistry:
    """Declarative singleton registry mapping canonical ``asset_class`` keys to ``AssetSessionPolicy`` instances."""

    _policies: Dict[str, AssetSessionPolicy] = {}

    @classmethod
    def register(cls, policy: AssetSessionPolicy) -> None:
        cls._policies[policy.asset_class] = policy

    @classmethod
    def get(cls, asset_class: str) -> AssetSessionPolicy:
        return cls._policies.get(asset_class) or cls._policies["crypto"]


_CRYPTO_24X7_POLICY = Crypto24x7SessionPolicy()
_TOKENIZED_EQUITY_24X7_POLICY = TokenizedEquity24x7SessionPolicy()
_EQUITY_RTH_POLICY = EquityRthSessionPolicy()

for _policy_inst in (_CRYPTO_24X7_POLICY, _TOKENIZED_EQUITY_24X7_POLICY, _EQUITY_RTH_POLICY):
    AssetClassPolicyRegistry.register(_policy_inst)


def resolve_asset_session_policy(symbol: str) -> AssetSessionPolicy:
    """Resolve the polymorphic ``AssetSessionPolicy`` (Crypto, TokenizedEquity, or RthEquity) for ``symbol``."""
    ident = resolve_symbol_identity(symbol)
    return AssetClassPolicyRegistry.get(ident.asset_class)
