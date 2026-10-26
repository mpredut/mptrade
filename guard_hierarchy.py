"""guard_hierarchy.py — Canonical Object-Oriented Guard Hierarchy & SSOT Registry.

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
from typing import Any


class BaseGuard(ABC):
    """Abstract base class for all order placement guards."""

    code: str
    canonical: str
    category: str
    aliases: set[str]
    description: str

    @staticmethod
    def _coerce_bypass_set(bypassed_guards: Any) -> set[str]:
        if not bypassed_guards:
            return set()
        if bypassed_guards is True:
            return {"all"}
        if isinstance(bypassed_guards, str):
            return {p.strip().lower() for p in bypassed_guards.split(",") if p.strip()}
        if isinstance(bypassed_guards, (set, frozenset, list, tuple)):
            out: set[str] = set()
            for x in bypassed_guards:
                if x is True:
                    out.add("all")
                elif x and not isinstance(x, bool):
                    for p in str(x).split(","):
                        ps = p.strip().lower()
                        if ps:
                            out.add(ps)
            return out
        return set()

    def is_bypassed(self, bypassed_guards: set[str]) -> bool:
        """Return whether this guard is bypassed by the given set of active bypass keys."""
        low = self._coerce_bypass_set(bypassed_guards)
        if not low:
            return False
        cat_aliases = {self.category}
        if self.category == "operational":
            cat_aliases.add("go")
        elif self.category == "trend":
            cat_aliases.add("gt")
        elif self.category == "profit":
            cat_aliases.update({"g_prof", "gf", "financial", "profit_guard"})
        elif self.category == "qlimit":
            cat_aliases.update({"g_qlimit", "gf", "financial"})
            if self.code == "G_QLIMIT1":
                cat_aliases.add("profit_guard")
        elif self.category == "intelligence":
            cat_aliases.update({"gi", "microstructure"})
        return bool(
            "all" in low
            or "*" in low
            or bool(low.intersection(cat_aliases))
            or self.code.lower() in low
            or self.canonical in low
            or bool(low.intersection(self.aliases))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "canonical": self.canonical,
            "category": self.category,
            "aliases": sorted(list(self.aliases)),
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
    description = "Symmetric 24h filled order cap (25/24h) and 5s anti-spam pause per symbol and side"


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


# Keep FinancialGuard as a base alias for type checks where needed
FinancialGuard = (ProfitGuard, QuantLimitGuard)


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

    def is_bypassed(self, bypassed_guards: set[str]) -> bool:
        low = self._coerce_bypass_set(bypassed_guards)
        if not low:
            return False
        gllm1_specific = {"gllm1", "llm_high_stake", "high_stake", "ai"}
        llm_cat = "gllm" in low or ("llm" in low and not bool(low.intersection(gllm1_specific)))
        return bool(
            "all" in low
            or "*" in low
            or llm_cat
            or self.code.lower() in low
            or self.canonical in low
            or bool(low.intersection(self.aliases))
        )


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

GUARD_CATALOG: dict[str, dict[str, Any]] = {
    g.code: {
        "code": g.code,
        "canonical": g.canonical,
        "aliases": g.aliases,
        "description": g.description,
        "category": g.category,
    }
    for g in GUARD_INSTANCES
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

    raw_tokens: list[str] = []
    if bypass_guards is not None:
        if bypass_guards is True or bypass_guards == "all" or bypass_guards == "*":
            raw_tokens.append("all")
        elif isinstance(bypass_guards, str):
            for part in bypass_guards.split(","):
                p = part.strip().lower()
                if p and p not in ("none", "off", "false", "no", "0"):
                    raw_tokens.append(p)
        elif isinstance(bypass_guards, (set, frozenset, list, tuple)):
            for item in bypass_guards:
                if item is True:
                    raw_tokens.append("all")
                elif item and not isinstance(item, bool):
                    for part in str(item).split(","):
                        p = part.strip().lower()
                        if p and p not in ("none", "off", "false", "no", "0"):
                            raw_tokens.append(p)

    for token in raw_tokens:
        if token in _ALL_GUARD_ALIASES:
            bypassed.add(token)
        else:
            print(
                f"[GUARD] Warning: unrecognized bypass guard token {token!r} ignored; "
                f"known guards are {', '.join(g.code for g in GUARD_INSTANCES)}"
            )

    # Expand master bypass ('all' or '*')
    if "all" in bypassed or "*" in bypassed:
        bypassed.update({
            "all", "*", "profit_guard", "operational", "trend",
            "profit", "qlimit", "financial", "intelligence", "llm", "gllm",
        })
        for g in GUARD_INSTANCES:
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    # Distinguish category-wide 'llm' / 'gllm' from GLLM1's 'llm' alias when re-normalizing
    gllm1_specific = {"gllm1", "llm_high_stake", "high_stake", "ai"}
    llm_category_requested = "gllm" in bypassed or (
        "llm" in bypassed and not bool(bypassed.intersection(gllm1_specific))
    )

    # Expand category groups and short prefixes
    category_shortcuts = {
        "go": "operational",
        "gt": "trend",
        "g_qlimit": "qlimit",
        "gf": "financial",
        "gi": "intelligence",
        "microstructure": "intelligence",
        "gllm": "llm",
    }
    for sc, cat_name in category_shortcuts.items():
        if sc in bypassed:
            bypassed.add(cat_name)

    for cat_name, cat_guards in GUARD_CATEGORIES.items():
        if cat_name == "llm" and not llm_category_requested:
            continue
        if cat_name in bypassed:
            if cat_name == "llm":
                bypassed.add("gllm")
            for g in cat_guards:
                bypassed.update(g.aliases)
                bypassed.add(g.canonical)
                bypassed.add(g.code.lower())

    # Expand composite profit_guard (G_PROF: math_profit + G_QLIMIT1: quantity_policy)
    if "profit_guard" in bypassed:
        bypassed.add("profit_guard")
        for code in ("G_PROF", "G_QLIMIT1"):
            g = GUARD_REGISTRY[code]
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    # Expand any individual token to include its canonical name and all alias variations
    for g in GUARD_INSTANCES:
        if bypassed.intersection(g.aliases) or g.code.lower() in bypassed:
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    return bypassed
