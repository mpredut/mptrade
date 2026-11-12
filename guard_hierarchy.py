"""guard_hierarchy.py -- Canonical Object-Oriented Guard Hierarchy & SSOT Registry.

Implements the Separation of Concerns (SoC) for trading safety, quantitative limits,
market microstructure intelligence, and LLM guards with mnemonic codes:
1. OperationalGuards (GO1): Execution velocity, volume caps, and anti-spam filters.
2. TrendGuards (GT_S, GT_M, GT_L): Multi-horizon trend deferrals and regime gates (short, medium, long).
3. ProfitGuards (G_PROF): Deterministic historical profit margin reference check.
4. QuantLimitGuards (G_QLIMIT1, G_QLIMIT2): 24h Gaussian weight limit and internal Parabolic/Weibull limits.
5. IntelligenceGuards (GI_OBOOK, GI_WHALE, GI_FUNDING): External market microstructure (Orderbook, Whales, Funding).
6. LLMGuards (GLLM1, GLLM2, GLLM3): Live High-Stake/Post-Stop LLM, 3h Sentiment Advisor LLM, and 4.5h Geopolitical LLM.

Guards are opt-out: any guard not explicitly bypassed executes with its default behavior.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

from botcore import (
    bool_env,
    normalize_side,
    parse_dotenv,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    split_csv,
)
from cachepaths import normalize_venue_name


_CATEGORY_SHORTCUTS: dict[str, str] = {
    "go": "operational",
    "gt": "trend",
    "g_prof": "profit",
    "g_qlimit": "qlimit",
    "gf": "financial",
    "gi": "intelligence",
    "microstructure": "intelligence",
    "gllm": "llm",
}

_GLLM1_SPECIFIC_ALIASES: frozenset[str] = frozenset({"gllm1", "llm_high_stake", "high_stake", "ai"})
_FALSY_BYPASS_TOKENS: frozenset[str] = frozenset({"none", "off", "false", "no", "0", "n", "f", "null", "nan"})
_CATEGORY_BYPASS_ALIASES: dict[str, frozenset[str]] = {
    "operational": frozenset({"operational", "go"}),
    "trend": frozenset({"trend", "gt"}),
    "profit": frozenset({"profit", "g_prof", "gf", "financial", "profit_guard"}),
    "qlimit": frozenset({"qlimit", "g_qlimit", "gf", "financial"}),
    "intelligence": frozenset({"intelligence", "gi", "microstructure"}),
}


def is_master_bypass(bypassed: Any) -> bool:
    """Return True when ``bypassed`` contains a master bypass token ('all' or '*')."""
    return bool(bypassed and ("all" in bypassed or "*" in bypassed))


def _is_llm_category_bypass(tokens: set[str]) -> bool:
    return "gllm" in tokens or ("llm" in tokens and not bool(tokens.intersection(_GLLM1_SPECIFIC_ALIASES)))


class BaseGuard(ABC):
    """Abstract base class for all order placement guards."""

    code: str
    canonical: str
    category: str
    aliases: set[str]
    description: str

    @staticmethod
    def _extract_raw_bypass_tokens(bypassed_guards: Any) -> list[str]:
        if not bypassed_guards:
            return []
        if bypassed_guards is True or bypassed_guards in ("all", "*"):
            return ["all"]
        if isinstance(bypassed_guards, BaseGuard):
            return [bypassed_guards.code.lower()]
        items = (
            [bypassed_guards]
            if isinstance(bypassed_guards, str)
            else (bypassed_guards if isinstance(bypassed_guards, (set, frozenset, list, tuple)) else ())
        )
        out: list[str] = []
        for x in items:
            if x is True:
                out.append("all")
            elif isinstance(x, BaseGuard):
                out.append(x.code.lower())
            elif x and not isinstance(x, bool):
                for ps in split_csv(str(x).replace(";", ","), lower=True):
                    if ps not in _FALSY_BYPASS_TOKENS:
                        out.append(ps)
        return out

    def add_tokens_to(self, target: set[str]) -> None:
        """Add this guard's lowercase code, canonical name, and aliases to ``target``."""
        target.update(self.aliases)
        target.add(self.canonical)
        target.add(self.code.lower())

    def _is_category_or_master_bypassed(self, low: set[str]) -> bool:
        cat_aliases = _CATEGORY_BYPASS_ALIASES.get(self.category, frozenset({self.category}))
        return bool(
            is_master_bypass(low)
            or bool(low.intersection(cat_aliases))
            or (self.code == "G_QLIMIT1" and "profit_guard" in low)
        )

    def is_bypassed(self, bypassed_guards: set[str]) -> bool:
        """Return whether this guard is bypassed by the given set of active bypass keys."""
        low = set(BaseGuard._extract_raw_bypass_tokens(bypassed_guards))
        if not low:
            return False
        return bool(
            self._is_category_or_master_bypassed(low)
            or self.code.lower() in low
            or self.canonical in low
            or bool(low.intersection(self.aliases))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "canonical": self.canonical,
            "category": self.category,
            "aliases": sorted(self.aliases),
            "description": self.description,
        }


# ============================================================================
# Category 1: Operational & Timing Guards (GO)
# ============================================================================

class OperationalGuard(BaseGuard):
    """Guards governing execution velocity, volume caps, and rate limits."""
    category = "operational"


class DailyLimitGuard(OperationalGuard):
    code = "GO1"
    canonical = "daily_limit"
    aliases = {"go1", "daily_limit"}
    description = "Symmetric 24h filled order cap (30/24h) per symbol and side"


# ============================================================================
# Category 2: Trend & Momentum Guards (GT_S, GT_M, GT_L)
# ============================================================================

class TrendGuard(BaseGuard):
    """Guards governing multi-horizon trend alignment, momentum deferral, and regime gates."""
    category = "trend"


class ShortTrendGuard(TrendGuard):
    code = "GT_S"
    canonical = "short_trend"
    aliases = {"gt_s", "short_trend", "trend_short"}
    description = "Short-term trend & momentum deferral via short_trend_manager (3.7m - 2.5h)"


class MediumTrendGuard(TrendGuard):
    code = "GT_M"
    canonical = "medium_trend"
    aliases = {"gt_m", "medium_trend", "trend_medium"}
    description = "Medium-term trend & market regime filter (4h - 24h)"


class LongTrendGuard(TrendGuard):
    code = "GT_L"
    canonical = "long_trend"
    aliases = {"gt_l", "long_trend", "trend_long", "macro_trend"}
    description = "Long-term macro trend validation from priceAnalysis (24h - 30 days)"


# ============================================================================
# Category 3: Deterministic Profit Reference Guard (G_PROF)
# ============================================================================

class ProfitGuard(BaseGuard):
    """Guards governing deterministic historical pricing profitability."""
    category = "profit"


class MathProfitGuard(ProfitGuard):
    code = "G_PROF"
    canonical = "math_profit"
    aliases = {"g_prof", "math_profit", "math_reference", "profit_margin"}
    description = "Deterministic historical profit margin reference check"


# ============================================================================
# Category 4: Internal Quantitative Limit Guards (G_QLIMIT1, G_QLIMIT2)
# ============================================================================

class QuantLimitGuard(BaseGuard):
    """Guards governing internal quantitative exposure sizing, parabolic spikes, and Weibull exhaustion."""
    category = "qlimit"


class QuantityPolicyGuard(QuantLimitGuard):
    code = "G_QLIMIT1"
    canonical = "quantity_policy"
    aliases = {"g_qlimit1", "quantity_policy", "weight_limit", "weight", "weight_sizing"}
    description = "24h cumulative risk-weighted quantity sizing via Gaussian curve and empirical T AUTO"


class InternalQuantLimitGuard(QuantLimitGuard):
    code = "G_QLIMIT2"
    canonical = "internal_quant_limit"
    aliases = {"g_qlimit2", "internal_quant_limit", "parabolic_surge", "weibull_exhaustion", "quantitative"}
    description = "Internal quantitative per-order limits: 2h Parabolic Surge and late-tail Weibull exhaustion"


# ============================================================================
# Category 5: External Market Microstructure Intelligence Guards (GI_*)
# ============================================================================

class IntelligenceGuard(BaseGuard):
    """Guards evaluating external market microstructure (Orderbook, Whale OI/Flow, Funding)."""
    category = "intelligence"


class OrderbookWallGuardSpec(IntelligenceGuard):
    code = "GI_OBOOK"
    canonical = "orderbook_wall"
    aliases = {"gi_obook", "orderbook_wall", "orderbook", "obook"}
    description = "External Microstructure: Spot Orderbook depth imbalance and Ask/Bid whale walls"


class WhaleFlowGuardSpec(IntelligenceGuard):
    code = "GI_WHALE"
    canonical = "whale_flow"
    aliases = {"gi_whale", "whale_flow", "whale_divergence", "whale"}
    description = "External Microstructure: Whale Top-Trader L/S ratio, Taker flow, and OI cascades"


class FundingCrowdingGuardSpec(IntelligenceGuard):
    code = "GI_FUNDING"
    canonical = "funding_crowding"
    aliases = {"gi_funding", "gi_crowd", "funding_crowding", "funding", "crowd"}
    description = "External Microstructure: Perpetual derivatives funding rate crowding"


# ============================================================================
# Category 6: LLM Reasoning & Macro Threat Guards (GLLM1, GLLM2, GLLM3)
# ============================================================================

class LLMGuard(BaseGuard):
    """Guards powered by LLM reasoning (live pre-trade, 3h sentiment advisor, 4.5h geopolitical)."""
    category = "llm"

    def _is_category_or_master_bypassed(self, low: set[str]) -> bool:
        return bool("all" in low or "*" in low or _is_llm_category_bypass(low))


class LLMHighStakeGuardSpec(LLMGuard):
    code = "GLLM1"
    canonical = "llm_high_stake"
    aliases = {"gllm1", "llm_high_stake", "high_stake", "ai", "llm"}
    description = "Live pre-trade LLM validation for high-stake orders (>= 700 USD) and post-stop re-entry"


class LLMSentimentGuardSpec(LLMGuard):
    code = "GLLM2"
    canonical = "llm_sentiment"
    aliases = {"gllm2", "llm_sentiment", "sentiment_advisor"}
    description = "Periodic 3h LLM Macro Sentiment Advisor (Fear & Greed, Market Breadth, Crypto News)"


class LLMGeopoliticalGuardSpec(LLMGuard):
    code = "GLLM3"
    canonical = "llm_geopolitical"
    aliases = {"gllm3", "llm_geopolitical", "geopolitical"}
    description = "Periodic 4.5h LLM Geopolitical & Energy Black Swan shock shield"


# ============================================================================
# SSOT Registries & Lookup Mappings
# ============================================================================

GUARD_INSTANCES: tuple[BaseGuard, ...] = (
    DailyLimitGuard(),
    ShortTrendGuard(),
    MediumTrendGuard(),
    LongTrendGuard(),
    MathProfitGuard(),
    QuantityPolicyGuard(),
    InternalQuantLimitGuard(),
    OrderbookWallGuardSpec(),
    WhaleFlowGuardSpec(),
    FundingCrowdingGuardSpec(),
    LLMHighStakeGuardSpec(),
    LLMSentimentGuardSpec(),
    LLMGeopoliticalGuardSpec(),
)

# Registry index keyed strictly by canonical codes (no legacy G1..G10 codes).
GUARD_REGISTRY: dict[str, BaseGuard] = {g.code: g for g in GUARD_INSTANCES}
# Also allow lookup by uppercase alias when it starts with GI_ (e.g. GI_CROWD -> GI_FUNDING)
GUARD_REGISTRY["GI_CROWD"] = GUARD_REGISTRY["GI_FUNDING"]

GUARD_CATEGORIES: dict[str, list[BaseGuard]] = {
    "operational": [g for g in GUARD_INSTANCES if isinstance(g, OperationalGuard)],
    "trend": [g for g in GUARD_INSTANCES if isinstance(g, TrendGuard)],
    "profit": [g for g in GUARD_INSTANCES if isinstance(g, ProfitGuard)],
    "qlimit": [g for g in GUARD_INSTANCES if isinstance(g, QuantLimitGuard)],
    "financial": [g for g in GUARD_INSTANCES if isinstance(g, (ProfitGuard, QuantLimitGuard))],
    "intelligence": [g for g in GUARD_INSTANCES if isinstance(g, IntelligenceGuard)],
    "llm": [g for g in GUARD_INSTANCES if isinstance(g, LLMGuard)],
}

_ALL_GUARD_ALIASES: set[str] = set().union(*(g.aliases for g in GUARD_INSTANCES)) | {
    "all", "*", "profit_guard",
    "operational", "trend", "profit", "qlimit", "financial", "intelligence", "microstructure", "llm",
    "go", "gt", "g_prof", "g_qlimit", "gf", "gi", "gllm",
}


def normalize_bypassed_guards(
    bypass_guards=None,
) -> set[str]:
    """Canonical resolver for bypassed guard identifiers (SSOT).

    Supports fine-grained guard opt-out: any guard not in the returned set
    executes with its standard default behavior.

    Supported inputs:
      - Canonical codes: GO1, GT_S, GT_M, GT_L, G_PROF, G_QLIMIT1, G_QLIMIT2,
        GI_OBOOK, GI_WHALE, GI_FUNDING, GLLM1, GLLM2, GLLM3
      - Canonical names: daily_limit, short_trend, medium_trend, long_trend,
        math_profit, quantity_policy, internal_quant_limit, orderbook_wall,
        whale_flow, funding_crowding, llm_high_stake, llm_sentiment, llm_geopolitical
      - Category groups: 'operational' ('go'), 'trend' ('gt'), 'profit', 'qlimit' ('g_qlimit'),
        'intelligence' ('gi'), 'llm' ('gllm')
      - Composite aliases: 'profit_guard' (bypasses G_PROF + G_QLIMIT1)
      - Master bypass: 'all' or '*' (bypasses all guards for emergency exits)
    """
    bypassed: set[str] = set()
    for token in BaseGuard._extract_raw_bypass_tokens(bypass_guards):
        if token in _ALL_GUARD_ALIASES:
            bypassed.add(token)
        else:
            print(
                f"[GUARD] Warning: unrecognized bypass guard token {token!r} ignored; "
                f"known guards are {', '.join(g.code for g in GUARD_INSTANCES)}"
            )

    # Expand master bypass ('all' or '*')
    if is_master_bypass(bypassed):
        bypassed.update({
            "all", "*", "profit_guard", "operational", "trend",
            "profit", "qlimit", "financial", "intelligence", "llm", "gllm",
        })
        for g in GUARD_INSTANCES:
            g.add_tokens_to(bypassed)

    # Distinguish category-wide 'llm' / 'gllm' from GLLM1's 'llm' alias when re-normalizing
    llm_category_requested = _is_llm_category_bypass(bypassed)

    # Expand category groups and short prefixes
    for sc, cat_name in _CATEGORY_SHORTCUTS.items():
        if sc in bypassed:
            bypassed.add(cat_name)

    for cat_name, cat_guards in GUARD_CATEGORIES.items():
        if cat_name == "llm" and not llm_category_requested:
            continue
        if cat_name in bypassed:
            if cat_name == "llm":
                bypassed.add("gllm")
            for g in cat_guards:
                g.add_tokens_to(bypassed)

    # Expand composite profit_guard (G_PROF: math_profit + G_QLIMIT1: quantity_policy)
    if "profit_guard" in bypassed:
        bypassed.add("profit_guard")
        for code in ("G_PROF", "G_QLIMIT1"):
            GUARD_REGISTRY[code].add_tokens_to(bypassed)

    # Expand any individual token to include its canonical name and all alias variations
    for g in GUARD_INSTANCES:
        if bypassed.intersection(g.aliases) or g.code.lower() in bypassed:
            g.add_tokens_to(bypassed)

    return bypassed


def revoke_guard_bypass(bypassed: set[str], code: str) -> None:
    """Remove a specific guard from an expanded bypass set while preserving sibling category tokens."""
    if not isinstance(bypassed, set) or not code or isinstance(code, bool):
        return
    code_str = str(code).strip()
    g = GUARD_REGISTRY.get(code_str.upper())
    if g is None:
        code_low = code_str.lower()
        g = next(
            (inst for inst in GUARD_INSTANCES if inst.canonical == code_low or code_low in inst.aliases),
            None,
        )
    if g is None:
        return

    # Normalize existing tokens in-place to lowercase so case-insensitive matches are cleanly removed
    lowered_existing = {str(t).strip().lower() for t in bypassed if t is not None and not isinstance(t, bool)}
    bypassed.clear()
    bypassed.update(lowered_existing)

    # 1. If master bypass ('all' or '*') was active, expand all sibling guards before revoking target
    if bypassed.intersection({"all", "*"}):
        for s in GUARD_INSTANCES:
            if s.code != g.code:
                s.add_tokens_to(bypassed)
        bypassed.difference_update({"all", "*"})

    # 2. If composite 'profit_guard' (G_PROF + G_QLIMIT1) was active and target is one of them, preserve sibling
    if g.code in ("G_PROF", "G_QLIMIT1") and "profit_guard" in bypassed:
        sibling_spec = GUARD_REGISTRY["G_QLIMIT1"] if g.code == "G_PROF" else GUARD_REGISTRY["G_PROF"]
        sibling_spec.add_tokens_to(bypassed)
        bypassed.discard("profit_guard")

    # 3. If composite 'financial' ('gf') category was active, preserve sibling financial guards
    if g in GUARD_CATEGORIES.get("financial", ()):
        if bypassed.intersection({"financial", "gf"}):
            for s in GUARD_CATEGORIES["financial"]:
                if s.code != g.code:
                    s.add_tokens_to(bypassed)
        bypassed.difference_update({"financial", "gf"})

    # 4. If target's category bypass was active, preserve sibling guards in that category
    cat_shortcuts = {sc for sc, cat in _CATEGORY_SHORTCUTS.items() if cat == g.category}
    cat_tokens = {g.category} | cat_shortcuts
    cat_active = _is_llm_category_bypass(bypassed) if g.category == "llm" else bool(bypassed.intersection(cat_tokens))
    if cat_active:
        for s in GUARD_CATEGORIES.get(g.category, ()):
            if s.code != g.code:
                s.add_tokens_to(bypassed)
    bypassed.difference_update(g.aliases)
    bypassed.discard(g.canonical)
    bypassed.discard(g.code.lower())
    if g.category == "llm":
        bypassed.discard("gllm")
        if cat_active or g.code == "GLLM1":
            bypassed.discard("llm")
    else:
        bypassed.difference_update(cat_tokens)


def is_category_bypassed(category: str, bypassed: set[str]) -> bool:
    """Return True when every guard in the given category is bypassed."""
    cat_norm = "" if isinstance(category, bool) else str(category or "").strip().lower()
    cat_key = _CATEGORY_SHORTCUTS.get(cat_norm, cat_norm)
    guards = GUARD_CATEGORIES.get(cat_key)
    return bool(guards) and all(g.is_bypassed(bypassed) for g in guards)


_QLIMIT2_GROUP_TOKENS: frozenset[str] = frozenset({
    "g_qlimit2",
    "internal_quant_limit",
    "quantitative",
    "qlimit",
    "g_qlimit",
    "gf",
    "financial",
    "all",
    "*",
})
_QLIMIT2_BASE_TOKENS: frozenset[str] = frozenset({
    "g_qlimit2",
    "internal_quant_limit",
    "quantitative",
})
_QLIMIT2_PARABOLIC_TOKENS: frozenset[str] = frozenset({
    "parabolic_surge",
    "parabolic_guard",
    "parabolicsurgeguard",
})
_QLIMIT2_WEIBULL_TOKENS: frozenset[str] = frozenset({
    "weibull_exhaustion",
    "exhaustion_guard",
    "weibullexhaustionguard",
})


def resolve_qlimit2_sub_bypasses(bypass_guards: object) -> tuple[set[str], bool, bool]:
    """Return ``(raw_bypass_tokens, parabolic_only_bypass, weibull_only_bypass)`` for ``G_QLIMIT2``."""
    raw_bypass_tokens = set(BaseGuard._extract_raw_bypass_tokens(bypass_guards)) if bypass_guards else set()
    has_group = bool(raw_bypass_tokens & _QLIMIT2_GROUP_TOKENS)
    has_parabolic = bool(raw_bypass_tokens & _QLIMIT2_PARABOLIC_TOKENS)
    has_weibull = bool(raw_bypass_tokens & _QLIMIT2_WEIBULL_TOKENS)
    parabolic_only = has_parabolic and not has_weibull and not has_group
    weibull_only = has_weibull and not has_parabolic and not has_group
    return raw_bypass_tokens, parabolic_only, weibull_only


def normalize_intel_bypass_with_qlimit2_subs(bypass_guards: Any) -> tuple[set[str], bool, bool]:
    """Normalize ``bypass_guards`` while preserving isolated Parabolic/Weibull sub-bypasses for ``G_QLIMIT2``."""
    _, parabolic_only, weibull_only = resolve_qlimit2_sub_bypasses(bypass_guards)
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards)
    if parabolic_only:
        bypassed.difference_update(_QLIMIT2_BASE_TOKENS | _QLIMIT2_WEIBULL_TOKENS)
        bypassed.add("parabolic_surge")
    elif weibull_only:
        bypassed.difference_update(_QLIMIT2_BASE_TOKENS | _QLIMIT2_PARABOLIC_TOKENS)
        bypassed.add("weibull_exhaustion")
    return bypassed, parabolic_only, weibull_only


_RAW_INTEL_CACHE_CODES: tuple[str, ...] = (
    "GT_L",
    "G_QLIMIT1",
    "G_QLIMIT2",
    "GI_OBOOK",
    "GI_WHALE",
    "GI_FUNDING",
    "GLLM2",
    "GLLM3",
)


def make_intel_cache_key(side: str, bypassed: set[str]) -> tuple[str, frozenset[str]]:
    """Build the canonical (side, bypassed_intel_codes) cache key for MarketRegimeContext."""
    side_norm = str(side or "").strip().upper() if not isinstance(side, bool) else ""
    _, parabolic_only, weibull_only = resolve_qlimit2_sub_bypasses(bypassed)
    codes = {
        code
        for code in _RAW_INTEL_CACHE_CODES
        if GUARD_REGISTRY[code].is_bypassed(bypassed)
        and not (code == "G_QLIMIT2" and (parabolic_only or weibull_only))
    }
    if parabolic_only:
        codes.add("G_QLIMIT2:PARABOLIC")
    elif weibull_only:
        codes.add("G_QLIMIT2:WEIBULL")
    return (side_norm, frozenset(codes))


# ============================================================================
# Canonical order_guard.conf Schema, Hot-Reload Loader & Pure Policy Helpers
# ============================================================================

_MARGINS: Optional[Dict[str, Any]] = None
_MARGINS_FILE_MTIME: float = 0.0
_MARGINS_LAST_CHECK: float = 0.0
_DEFAULT_ORDER_GUARD_ANCHOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order_guard.py")

_REQUIRED_ORDER_GUARD_KEYS = frozenset({
    "default", "binance", "kraken", "default_window_h", "kraken_window_h",
    "default_weight_proxy", "binance_weight_proxy", "kraken_weight_proxy",
    "default_chop_weight", "binance_chop_weight", "kraken_chop_weight",
    "default_max_daily_trades", "default_safeback_sec",
    "default_buy_reference", "binance_buy_reference", "kraken_buy_reference",
    "default_sell_reference", "binance_sell_reference", "kraken_sell_reference",
    "assetguardian_buy_reference", "assetguardian_sell_reference",
    "monitortrades_buy_reference", "monitortrades_sell_reference",
    "tradeall_buy_reference", "tradeall_sell_reference",
    "rtrade_buy_reference", "rtrade_sell_reference",
    "dynamic_window_favorable_h", "dynamic_window_flat_h", "dynamic_window_adverse_h",
    "regime_context_max_age_sec",
    # Pillar 1: Internal Quantitative Mathematical Guards
    "internal_guard_mode", "intelligence_guards_mode", "parabolic_surge_pct",
    "parabolic_pullback_pct", "weibull_exhaustion_policy", "weibull_exhausted_scale",
    "weibull_p90_days",
    # Pillar 2: External Microstructure & Derivatives Flow Guards
    "external_guard_mode", "whale_guard_mode", "severe_cascade_oi_pct", "min_taker_ratio",
    "taker_downscale_threshold", "taker_downscaled_scale", "orderbook_wall_guard_mode",
    "whale_wall_usd_limit", "min_buy_imbalance", "max_sell_imbalance",
    "high_stake_min_notional_usd", "funding_guard_mode", "funding_max_long_rate",
    "funding_min_short_rate", "funding_crowding_policy", "funding_crowded_scale",
    # Pillar 2 Sub-Module: Smart Stop-Loss
    "smart_sl_mode", "smart_sl_max_threshold_pct", "smart_sl_grace_seconds",
    "smart_sl_disaster_buffer_pct", "smart_sl_min_whale_long_pct", "smart_sl_min_taker_ratio",
    # Pillar 3: AI Intelligence High-Stake Guard & Post-Stop Re-entry
    "llm_guard_mode", "llm_timeout_sec", "llm_fallback", "post_stop_reentry_mode",
    "post_stop_reentry_timeout_sec", "post_stop_reentry_fallback", "post_stop_reentry_notify",
    "post_stop_reentry_cache_ttl_sec", "macro_advisor_interval_sec", "macro_llm_interval_sec",
    # Pillar 4 & Global + Autonomous AI Supervisor & Trading Gateway + Bypass
    "geopolitical_guard_mode", "shadow_notify", "supervisor_mode", "supervisor_notify",
    "autonomous_trading_mode", "autonomous_trading_max_notional_usd",
    "autonomous_trading_min_notional_usd", "autonomous_trading_min_confidence",
    "autonomous_trading_notify", "default_bypass_guards", "binance_bypass_guards",
    "kraken_bypass_guards",
})

VALID_PROFIT_GUARD_MODES = frozenset({"strict", "average", "last", "hybrid", "dynamic", "off"})
VALID_GUARD_MODES = frozenset({"off", "shadow", "enforce"})

_GUARD_MODE_KEYS = (
    "internal_guard_mode", "intelligence_guards_mode", "external_guard_mode",
    "whale_guard_mode", "orderbook_wall_guard_mode", "funding_guard_mode",
    "smart_sl_mode", "llm_guard_mode", "post_stop_reentry_mode",
    "geopolitical_guard_mode", "supervisor_mode", "autonomous_trading_mode",
)

_MODULE_REFERENCE_KEYS = frozenset({
    "assetguardian_buy_reference", "assetguardian_sell_reference",
    "monitortrades_buy_reference", "monitortrades_sell_reference",
    "tradeall_buy_reference", "tradeall_sell_reference",
    "rtrade_buy_reference", "rtrade_sell_reference",
})

_DISCRETE_POLICY_SETS: Dict[str, frozenset] = {
    "weibull_exhaustion_policy": frozenset({"block", "veto", "downscale"}),
    "funding_crowding_policy": frozenset({"defer", "downscale"}),
    "llm_fallback": frozenset({"allow", "block", "veto", "downscale"}),
    "post_stop_reentry_fallback": frozenset({"allow", "block", "veto", "downscale"}),
}


def _validate_discrete_cfg(val: Any, valid_set: Any, label: str) -> str:
    if not isinstance(val, str):
        raise ValueError(f"Invalid {label} {val!r}; must be a string in {sorted(valid_set)}")
    mode = val.strip().lower()
    if mode not in valid_set:
        raise ValueError(f"Invalid {label} {val!r}; must be one of {sorted(valid_set)}")
    return mode


def validate_profit_guard_mode_cfg(val: Any) -> str:
    """Validate that a profit guard mode is strictly one of VALID_PROFIT_GUARD_MODES."""
    return _validate_discrete_cfg(val, VALID_PROFIT_GUARD_MODES, "profit guard mode")


def validate_guard_mode_cfg(val: Any, key_name: str = "guard_mode") -> str:
    """Validate that a guard / supervisor / gateway mode is strictly in VALID_GUARD_MODES."""
    return _validate_discrete_cfg(val, VALID_GUARD_MODES, key_name)


def is_intelligence_master_off(margins: Dict[str, Any], *, require_key: bool = False) -> bool:
    """Return True when intelligence_guards_mode is set to 'off'."""
    if not isinstance(margins, Mapping):
        if require_key:
            raise KeyError("intelligence_guards_mode")
        return False
    if not require_key and "intelligence_guards_mode" not in margins:
        return False
    return validate_guard_mode_cfg(margins["intelligence_guards_mode"], "intelligence_guards_mode") == "off"


def preflight_intelligence_guard_entry(
    side: Any,
    *,
    margins: Optional[Dict[str, Any]] = None,
    load_margins_fn: Any = None,
    bypass_guards: Any = None,
    pre_normalized_bypassed: Optional[set[str]] = None,
    guard_code: Optional[str] = None,
    guard_bypass_reason: str = "stop_loss_exempt",
    allowed_sides: Tuple[str, ...] = ("BUY", "SELL"),
    unsupported_side_reason: str = "not_supported_side",
    require_master_key: bool = False,
    check_master_before_side: bool = False,
) -> Tuple[Optional[Tuple[bool, str, float]], str, Dict[str, Any], set[str]]:
    """Run shared preflight for intelligence guard entrypoints.

    Returns ``(early_result, order_side, margins_dict, bypassed_set)`` where ``early_result``
    is ``(True, reason, 1.0)`` when bypassed, side-exempt, or master-disabled, else ``None``.
    """
    bypassed = (
        pre_normalized_bypassed
        if pre_normalized_bypassed is not None
        else (normalize_bypassed_guards(bypass_guards=bypass_guards) if bypass_guards else set())
    )
    if is_master_bypass(bypassed):
        return (True, "stop_loss_exempt", 1.0), "", {}, bypassed
    if guard_code and GUARD_REGISTRY[guard_code].is_bypassed(bypassed):
        return (True, guard_bypass_reason, 1.0), "", {}, bypassed
    loader = load_margins_fn if callable(load_margins_fn) else _load_margins
    if check_master_before_side:
        m = loader() if margins is None else margins
        if is_intelligence_master_off(m, require_key=require_master_key):
            return (True, "intelligence_guards_off", 1.0), "", m, bypassed
        order_side = normalize_side(side)
        if order_side not in allowed_sides:
            return (True, unsupported_side_reason, 1.0), order_side, m, bypassed
        return None, order_side, m, bypassed
    order_side = normalize_side(side)
    if order_side not in allowed_sides:
        return (True, unsupported_side_reason, 1.0), order_side, {}, bypassed
    m = loader() if margins is None else margins
    if is_intelligence_master_off(m, require_key=require_master_key):
        return (True, "intelligence_guards_off", 1.0), order_side, m, bypassed
    return None, order_side, m, bypassed


def _validate_margins_cfg(m: Dict[str, Any], path: str) -> None:
    """Validate discrete mode and policy values in order_guard.conf at load time."""
    for key in _GUARD_MODE_KEYS:
        validate_guard_mode_cfg(m[key], key_name=key)

    for key, val in m.items():
        if key.endswith("_buy_reference") or key.endswith("_sell_reference"):
            if key in _MODULE_REFERENCE_KEYS:
                sval = str(val).strip().lower() if isinstance(val, str) else None
                if sval != "venue":
                    validate_profit_guard_mode_cfg(val)
            else:
                validate_profit_guard_mode_cfg(val)

    for policy_key, valid_set in _DISCRETE_POLICY_SETS.items():
        _validate_discrete_cfg(m[policy_key], valid_set, f"{policy_key}")

    min_auto_usd = safe_pos_float(m.get("autonomous_trading_min_notional_usd"), default=None)
    max_auto_usd = safe_pos_float(m.get("autonomous_trading_max_notional_usd"), default=None)
    if min_auto_usd is None:
        raise ValueError(
            f"Invalid autonomous_trading_min_notional_usd {m.get('autonomous_trading_min_notional_usd')!r} in {path}"
        )
    if max_auto_usd is None or max_auto_usd < min_auto_usd:
        raise ValueError(
            f"Invalid autonomous_trading_max_notional_usd {m.get('autonomous_trading_max_notional_usd')!r} in {path}; "
            f"must be >= autonomous_trading_min_notional_usd ({min_auto_usd})"
        )

    fav_h = safe_pos_float(m.get("dynamic_window_favorable_h"), default=None)
    flat_h = safe_pos_float(m.get("dynamic_window_flat_h"), default=None)
    adv_h = safe_pos_float(m.get("dynamic_window_adverse_h"), default=None)
    if fav_h is None or flat_h is None or adv_h is None or not (fav_h <= flat_h <= adv_h):
        raise ValueError(
            f"Invalid dynamic window progression in {path}: require 0 < dynamic_window_favorable_h ({fav_h}) "
            f"<= dynamic_window_flat_h ({flat_h}) <= dynamic_window_adverse_h ({adv_h})"
        )

    pullback_pct = safe_pos_float(m.get("parabolic_pullback_pct"), default=None)
    surge_pct = safe_pos_float(m.get("parabolic_surge_pct"), default=None)
    if pullback_pct is None or surge_pct is None or pullback_pct > surge_pct:
        raise ValueError(
            f"Invalid parabolic bounds in {path}: require 0 < parabolic_pullback_pct ({pullback_pct}) "
            f"<= parabolic_surge_pct ({surge_pct})"
        )

    min_taker = safe_pos_float(m.get("min_taker_ratio"), default=None)
    downscale_taker = safe_pos_float(m.get("taker_downscale_threshold"), default=None)
    if min_taker is None or downscale_taker is None or min_taker > downscale_taker:
        raise ValueError(
            f"Invalid taker ratio bounds in {path}: require 0 < min_taker_ratio ({min_taker}) "
            f"<= taker_downscale_threshold ({downscale_taker})"
        )

    min_imb = safe_pos_float(m.get("min_buy_imbalance"), default=None)
    max_imb = safe_pos_float(m.get("max_sell_imbalance"), default=None)
    if min_imb is None or max_imb is None or not (0.0 < min_imb < max_imb < 1.0):
        raise ValueError(
            f"Invalid orderbook imbalance bounds in {path}: require 0 < min_buy_imbalance ({min_imb}) "
            f"< max_sell_imbalance ({max_imb}) < 1.0"
        )

    min_short_rate = safe_float(m.get("funding_min_short_rate"), default=None)
    max_long_rate = safe_pos_float(m.get("funding_max_long_rate"), default=None)
    if min_short_rate is None or max_long_rate is None or not (min_short_rate < 0.0 < max_long_rate):
        raise ValueError(
            f"Invalid funding rate bounds in {path}: require funding_min_short_rate ({min_short_rate}) "
            f"< 0 < funding_max_long_rate ({max_long_rate})"
        )


def _sync_margins_state(
    og_mod: Any,
    margins: Optional[Dict[str, Any]],
    file_mtime: float,
    last_check: float,
) -> Optional[Dict[str, Any]]:
    global _MARGINS, _MARGINS_FILE_MTIME, _MARGINS_LAST_CHECK
    _MARGINS = margins
    _MARGINS_FILE_MTIME = file_mtime
    _MARGINS_LAST_CHECK = last_check
    if og_mod is not None:
        og_mod._MARGINS = margins
        og_mod._MARGINS_FILE_MTIME = file_mtime
        og_mod._MARGINS_LAST_CHECK = last_check
    return margins


def _load_margins() -> Dict[str, Any]:
    """Read and cache ``key = value`` lines strictly from ``order_guard.conf``.

    The configuration file is the SINGLE source of truth without code fallbacks.
    Checks file mtime periodically (every 5 seconds) to hot-reload config edits,
    and transparently honors ``order_guard._load_margins`` / ``order_guard._MARGINS``
    overrides when ``order_guard`` is loaded in ``sys.modules``.
    """
    og_mod = sys.modules.get("order_guard")
    if og_mod is not None:
        patched_loader = og_mod.__dict__.get("_load_margins")
        if callable(patched_loader) and patched_loader is not _ORIG_LOAD_MARGINS:
            return patched_loader()
        has_og_state = "_MARGINS" in og_mod.__dict__
        cur_margins = og_mod.__dict__.get("_MARGINS") if has_og_state else _MARGINS
        cur_mtime = safe_nonneg_float(og_mod.__dict__.get("_MARGINS_FILE_MTIME", _MARGINS_FILE_MTIME), default=0.0) or 0.0
        cur_last_check = safe_nonneg_float(og_mod.__dict__.get("_MARGINS_LAST_CHECK", _MARGINS_LAST_CHECK), default=0.0) or 0.0
        anchor_file = og_mod.__dict__.get("__file__") or _DEFAULT_ORDER_GUARD_ANCHOR
    else:
        cur_margins = _MARGINS
        cur_mtime = _MARGINS_FILE_MTIME
        cur_last_check = _MARGINS_LAST_CHECK
        anchor_file = _DEFAULT_ORDER_GUARD_ANCHOR

    now_ts = time.time()
    path = os.path.join(os.path.dirname(os.path.abspath(anchor_file)), "order_guard.conf")
    if cur_margins is not None and 0.0 <= (now_ts - cur_last_check) < 5.0:
        return cur_margins
    _sync_margins_state(og_mod, cur_margins, cur_mtime, now_ts)
    if not os.path.exists(path):
        if cur_margins is not None:
            return cur_margins
        raise FileNotFoundError(f"Missing mandatory configuration file: {path}")
    mtime = os.path.getmtime(path)
    if cur_margins is not None and (cur_mtime == 0.0 or mtime == cur_mtime):
        return _sync_margins_state(og_mod, cur_margins, mtime, now_ts)  # type: ignore[return-value]
    m: Dict[str, Any] = parse_dotenv(path, lower_keys=True, coerce_floats=True)
    missing = sorted(_REQUIRED_ORDER_GUARD_KEYS - set(m.keys()))
    if missing:
        if cur_margins is not None and bool_env("BACKTEST_MODE", default=False):
            return _sync_margins_state(og_mod, cur_margins, mtime, now_ts)  # type: ignore[return-value]
        raise KeyError(f"Missing required keys in {path}: {', '.join(missing)}")
    _validate_margins_cfg(m, path)
    return _sync_margins_state(og_mod, m, mtime, now_ts)  # type: ignore[return-value]


_ORIG_LOAD_MARGINS = _load_margins


def resolve_high_stake_min_notional_usd(
    margins: Dict[str, Any],
    *,
    side: Optional[str] = None,
    require_key: bool = False,
) -> float:
    """Resolve the unified high-stake notional threshold (high_stake_min_notional_usd) from margins."""
    if isinstance(margins, dict) and "high_stake_min_notional_usd" in margins:
        return safe_pos_float(margins.get("high_stake_min_notional_usd"), default=700.0) or 700.0
    if require_key:
        raise KeyError("high_stake_min_notional_usd")
    return 700.0


def _venue_cfg(
    provider_name: Any,
    suffix: str = "",
    default: Any = None,
    *,
    margins: Optional[Dict[str, Any]] = None,
    required: bool = False,
) -> Any:
    """Resolve ``<venue><suffix>`` with fallback to ``default<suffix>`` from ``order_guard.conf``."""
    m = margins if isinstance(margins, dict) else _load_margins()
    venue = (normalize_venue_name(provider_name, default="") or str(provider_name or "")).strip().lower()
    venue_key = f"{venue}{suffix}" if venue else ""
    default_key = f"default{suffix}"
    if venue_key and venue_key in m:
        return m[venue_key]
    if default_key in m:
        return m[default_key]
    if required:
        raise KeyError(
            f"Missing required profit guard config key {venue_key or default_key!r} (and {default_key!r}) in order_guard.conf"
        )
    return default


def resolve_effective_bypass_guards(
    venue: Any = None,
    *,
    bypass_guards: Any = None,
    inst_bypass: Any = None,
    margins: Optional[Dict[str, Any]] = None,
) -> set[str]:
    """Return normalized bypass guards unioning venue config, instrument config, and caller overrides."""
    res = normalize_bypassed_guards(
        bypass_guards=_venue_cfg(venue, "_bypass_guards", margins=margins)
    )
    if inst_bypass:
        res |= normalize_bypassed_guards(bypass_guards=inst_bypass)
    if bypass_guards:
        res |= normalize_bypassed_guards(bypass_guards=bypass_guards)
    return res


def _is_valid_regime_context(
    regime_context: Any,
    symbol: Optional[str],
    provider: Any,
    *,
    margins: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
    require_identity: bool = True,
    default_max_age: float = 600.0,
) -> bool:
    """Return True when ``regime_context`` is non-None and passes identity/freshness validation."""
    if regime_context is None:
        return False
    if not hasattr(regime_context, "is_valid_for"):
        return True
    m = _load_margins() if margins is None else margins
    max_age = safe_pos_float(m.get("regime_context_max_age_sec") if isinstance(m, dict) else None, default_max_age)
    return bool(
        regime_context.is_valid_for(
            symbol=symbol,
            provider=provider,
            max_age_seconds=max_age,
            now=now,
            require_identity=require_identity,
        )
    )


def merge_guard_scale(
    effective_scale: float,
    active_reason: str,
    new_scale: float,
    new_reason: str,
) -> Tuple[float, str]:
    """Combine a candidate guard scale into ``(effective_scale, active_reason)`` keeping the strictest scale."""
    if new_scale < 1.0 and new_scale <= effective_scale:
        return new_scale, new_reason
    return effective_scale, active_reason


def set_regime_context_attrs(regime_context: Any, **attrs: Any) -> None:
    """Assign attributes onto a (potentially frozen) ``regime_context`` object via ``object.__setattr__``."""
    for attr_name, val in attrs.items():
        object.__setattr__(regime_context, attr_name, val)
