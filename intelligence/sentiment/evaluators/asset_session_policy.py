"""3-tier polymorphic AssetSessionPolicy hierarchy (Crypto, TokenizedEquity, RthEquity).

Level 1: ``AssetClassPolicyInterface`` (pure abstract contract).
Level 2: ``AssetSessionPolicy`` and ``Continuous24x7SessionPolicy`` (domain abstract bases).
Level 3: ``Crypto24x7SessionPolicy``, ``TokenizedEquity24x7SessionPolicy``,
         ``EquityRthSessionPolicy`` registered in ``AssetClassPolicyRegistry``.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from botcore import _prices_diverged, safe_nonneg_float, safe_pos_float
from instrument_registry import resolve_symbol_identity
from intelligence.macro_sentiment_ssot import (
    SHORT_KINEMATIC_METRIC_KEYS,
    parse_ts_px_row,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    CLOSED_EQUITY_INTRADAY_KEYS,
    CLOSED_EQUITY_QUOTE_AGE_MIN,
    STALE_SHORT_24H_KEYS,
    _RTH_ATR_ANCHOR_KEYS,
    _RTH_VP_ANCHOR_KEYS,
    _RTH_VWAP_ANCHOR_KEYS,
    _SHORT_SLOPE_OVERRIDE_KEYS,
    compute_24h_price_metrics,
    normalize_and_dedup_24h_series,
    pct_delta,
    prepend_older_non_diverged_points,
    sort_ts_rows,
    strip_off_rth_flat_poller_tail,
)


class AssetClassPolicyInterface(ABC):
    """Pure abstract contract (Level 1) for asset-class market hours, derivatives availability, and series normalization."""

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Canonical asset-class key ('crypto', 'tokenized_equity', 'rth_equity')."""

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
        return False

    @property
    def has_liquidation_clusters(self) -> bool:
        return False

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
        return prepend_older_non_diverged_points(raw_ser, ohlc24_pts)

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
    def has_crypto_perpetuals(self) -> bool:
        return True

    @property
    def has_liquidation_clusters(self) -> bool:
        return True

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
    def is_tokenized_equity(self) -> bool:
        return True


class EquityRthSessionPolicy(AssetSessionPolicy):
    """Concrete policy (Level 3) for RTH-gated cash equities (`_EQ` / `_US_EQ`)."""

    cls_name = "EquityRthSessionPolicy"

    @property
    def asset_class(self) -> str:
        return "rth_equity"

    @property
    def is_equity(self) -> bool:
        return True

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
        return strip_off_rth_flat_poller_tail(sort_ts_rows(rows), symbol, is_market_open_fn=is_market_open_fn)

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
    ) -> Tuple[list, Dict[str, Any], bool]:
        eff_compute_24h = compute_24h_fn or compute_24h_price_metrics
        s24_p0 = parse_ts_px_row(matched_series_24h[0]) if len(matched_series_24h) >= 2 else None
        s24_p1 = parse_ts_px_row(matched_series_24h[-1]) if len(matched_series_24h) >= 2 else None
        replaced = False
        override_px = round(cur_px, 4)
        if s24_p1 is not None and len(matched_series_24h) >= 2:
            rth_vp_anchors = {k_vp: m24[k_vp] for k_vp in _RTH_VP_ANCHOR_KEYS if k_vp in m24}
            rth_vwap_anchors = {k_vw: m24[k_vw] for k_vw in _RTH_VWAP_ANCHOR_KEYS if k_vw in m24}
            rth_atr_anchors = {k_at: m24[k_at] for k_at in _RTH_ATR_ANCHOR_KEYS if k_at in m24}
            last_bar_tail = (
                list(matched_series_24h[-1][2:])
                if isinstance(matched_series_24h[-1], (list, tuple)) and len(matched_series_24h[-1]) >= 3
                else []
            )
            override_series_24h = list(matched_series_24h[:-1]) + [[s24_p1[0], override_px, *last_bar_tail]]
            m24 = (
                eff_compute_24h(override_series_24h, is_equity=True)
                if eff_compute_24h is compute_24h_price_metrics
                else eff_compute_24h(override_series_24h)
            ) or m24
            if rth_vp_anchors:
                for k_vp_clr in _RTH_VP_ANCHOR_KEYS:
                    m24.pop(k_vp_clr, None)
                m24.update(rth_vp_anchors)
            if rth_vwap_anchors:
                for k_vw_clr in _RTH_VWAP_ANCHOR_KEYS:
                    m24.pop(k_vw_clr, None)
                m24.update(rth_vwap_anchors)
                if (vwap_px := safe_pos_float(m24.get("session_vwap_px"), default=None)) is not None:
                    m24["dist_vs_session_vwap_pct"] = pct_delta(override_px, vwap_px)
                    if (sig_pct := safe_pos_float(m24.get("session_vwap_sigma_pct"), default=None)) is not None:
                        sig_px = vwap_px * (sig_pct / 100.0)
                        if sig_px > 0.0:
                            m24["session_vwap_z"] = round((override_px - vwap_px) / sig_px, 2) + 0.0
                    else:
                        m24.pop("session_vwap_z", None)
            if rth_atr_anchors:
                for k_at_clr in _RTH_ATR_ANCHOR_KEYS:
                    m24.pop(k_at_clr, None)
                m24.update(rth_atr_anchors)
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
        if (
            m24
            and s24_p1 is not None
            and (
                (math.trunc(now_ts * 1000) - s24_p1[0]) > 20 * 60 * 1000
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
                    matched_series_24h, m24, sym_px_cur, entry=entry, compute_24h_fn=compute_24h_fn
                )
            else:
                for k_stale in STALE_SHORT_24H_KEYS:
                    m24.pop(k_stale, None)
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
        if not isinstance(entry, dict) or post_rth_series_replaced:
            return
        q_age_min = safe_nonneg_float(entry.get("quote_age_min"), default=None)
        if q_age_min is not None and q_age_min > CLOSED_EQUITY_QUOTE_AGE_MIN:
            for k_closed in CLOSED_EQUITY_INTRADAY_KEYS:
                m24.pop(k_closed, None)
                entry.pop(k_closed, None)
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
                and (math.trunc(now_ts * 1000) - last_eq_pt[0]) > 20 * 60 * 1000
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
