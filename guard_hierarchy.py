"""guard_hierarchy.py — Canonical Object-Oriented Guard Hierarchy & SSOT Registry.

Implements the Separation of Concerns (SoC) for trading safety and intelligence guards
with mnemonic G<LETTER><INDEX> encoding:
1. OperationalGuards (GO1): Execution velocity, volume caps, and anti-spam filters.
2. TrendGuards (GT1..GT3): Multi-horizon trend deferrals and regime gates (short, medium, long).
3. FinancialGuards (GF1, GF2): Mathematical price profitability and risk-weighted sizing.
4. IntelligenceGuards (GI1..GI4): Multi-pillar macro, micro, sentiment, and geopolitical threats.

Guards are opt-out: any guard not explicitly bypassed executes with its default behavior.
Legacy codes G1..G8 are fully supported as seamless backward-compatible aliases.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Optional, Set, Dict, Type


class BaseGuard(ABC):
    """Abstract base class for all order placement guards."""

    code: str
    canonical: str
    category: str
    aliases: set[str]
    description: str

    def is_bypassed(self, bypassed_guards: set[str]) -> bool:
        """Return whether this guard is bypassed by the given set of active bypass keys."""
        if not bypassed_guards:
            return False
        return bool(
            "all" in bypassed_guards
            or "*" in bypassed_guards
            or self.category in bypassed_guards
            or self.code.lower() in bypassed_guards
            or self.canonical in bypassed_guards
            or bool(bypassed_guards.intersection(self.aliases))
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
    aliases = {"go1", "g1", "daily_limit", "daily_cap"}
    description = "Daily trade count cap and anti-spam temporal filter"


# ============================================================================
# Category 2: Trend & Momentum Guards (GT)
# ============================================================================

class TrendGuard(BaseGuard):
    """Guards governing multi-horizon trend alignment, momentum deferral, and regime gates."""
    category = "trend"


class ShortTrendGuard(TrendGuard):
    code = "GT1"
    canonical = "short_trend"
    aliases = {"gt1", "g2", "short_trend", "trend_short"}
    description = "Short-term trend & momentum deferral via short_trend_manager (3.7m - 2.5h)"


class MediumTrendGuard(TrendGuard):
    code = "GT2"
    canonical = "medium_trend"
    aliases = {"gt2", "g9", "medium_trend", "trend_medium"}
    description = "Medium-term trend & market regime filter (4h - 24h)"


class LongTrendGuard(TrendGuard):
    code = "GT3"
    canonical = "long_trend"
    aliases = {"gt3", "g10", "long_trend", "trend_long", "macro_trend"}
    description = "Long-term macro trend validation from priceAnalysis (24h - 30 days)"


# ============================================================================
# Category 3: Financial & Capital Protection Guards (GF)
# ============================================================================

class FinancialGuard(BaseGuard):
    """Guards governing deterministic pricing profitability and capital exposure."""
    category = "financial"


class MathProfitGuard(FinancialGuard):
    code = "GF1"
    canonical = "math_profit"
    aliases = {"gf1", "g3", "math_profit", "math_reference", "profit_margin"}
    description = "Deterministic historical profit margin reference check"


class QuantityPolicyGuard(FinancialGuard):
    code = "GF2"
    canonical = "quantity_policy"
    aliases = {"gf2", "g4", "quantity_policy", "weight", "weight_sizing"}
    description = "Dynamic exposure and risk-weighted quantity sizing"


# ============================================================================
# Category 4: Market Intelligence & Threat Shield Guards (GI)
# ============================================================================

class IntelligenceGuard(BaseGuard):
    """Guards evaluating market microstructure, momentum exhaustion, AI, and geopolitics."""
    category = "intelligence"


class Pillar1QuantitativeGuard(IntelligenceGuard):
    code = "GI1"
    canonical = "p1_quantitative"
    aliases = {"gi1", "g5", "p1_quantitative", "p1", "pillar1", "quantitative"}
    description = "Pillar 1: Parabolic Surge and Weibull trend exhaustion"


class Pillar2MicrostructureGuard(IntelligenceGuard):
    code = "GI2"
    canonical = "p2_microstructure"
    aliases = {"gi2", "g6", "p2_microstructure", "p2", "pillar2", "microstructure"}
    description = "Pillar 2: Orderbook Ask Wall, Funding crowding, and Whale flow"


class Pillar3LLMGuard(IntelligenceGuard):
    code = "GI3"
    canonical = "p3_llm"
    aliases = {"gi3", "g7", "p3_llm", "p3", "pillar3", "ai", "llm"}
    description = "Pillar 3: High-stake pre-trade LLM validation"


class Pillar4GeopoliticalGuard(IntelligenceGuard):
    code = "GI4"
    canonical = "p4_geopolitical"
    aliases = {"gi4", "g8", "p4_geopolitical", "p4", "pillar4", "geopolitical"}
    description = "Pillar 4: Geopolitical Black Swan shock shield"


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
    Pillar1QuantitativeGuard(),
    Pillar2MicrostructureGuard(),
    Pillar3LLMGuard(),
    Pillar4GeopoliticalGuard(),
)

# Registry index keyed by both canonical codes (GO1, GT1..GT3, GF1..GF2, GI1..GI4)
# and legacy codes (G1..G8, G9, G10) for seamless backward compatibility.
GUARD_REGISTRY: dict[str, BaseGuard] = {}
for g in GUARD_INSTANCES:
    GUARD_REGISTRY[g.code] = g
    for al in g.aliases:
        if al.upper().startswith("G") and len(al) in (2, 3, 4):
            GUARD_REGISTRY.setdefault(al.upper(), g)

GUARD_CATEGORIES: dict[str, list[BaseGuard]] = {
    "operational": [g for g in GUARD_INSTANCES if isinstance(g, (OperationalGuard, ShortTrendGuard))],
    "trend": [g for g in GUARD_INSTANCES if isinstance(g, TrendGuard)],
    "financial": [g for g in GUARD_INSTANCES if isinstance(g, FinancialGuard)],
    "intelligence": [g for g in GUARD_INSTANCES if isinstance(g, IntelligenceGuard)],
}

# Catalog mapping matching both new codes and legacy G1..G8 schemas
GUARD_CATALOG: dict[str, dict[str, Any]] = {}
for g in GUARD_INSTANCES:
    d = {
        "code": g.code,
        "canonical": g.canonical,
        "aliases": g.aliases,
        "description": g.description,
        "category": g.category,
    }
    GUARD_CATALOG[g.code] = d
    for al in g.aliases:
        if al.upper().startswith("G") and len(al) in (2, 3, 4):
            GUARD_CATALOG.setdefault(al.upper(), d)

_ALL_GUARD_ALIASES: set[str] = set().union(*(g.aliases for g in GUARD_INSTANCES)) | {
    "all", "*", "profit_guard", "legacy_profit_guard", "cooldown",
    "operational", "trend", "financial", "intelligence",
    "go", "gt", "gf", "gi",
}


def normalize_bypassed_guards(
    bypass_guards=None,
    legacy_bypass_profit_guard=None,
    legacy_bypass_profit_reference=None,
    legacy_bypass_quantity_policy=None,
) -> set[str]:
    """Canonical resolver for bypassed guard identifiers (SSOT).

    Supports fine-grained guard opt-out: any guard not in the returned set
    executes with its standard default behavior.

    Supported inputs:
      - Canonical codes: GO1, GT1..GT3, GF1..GF2, GI1..GI4
      - Legacy codes: G1..G8, G9, G10
      - Canonical names: daily_limit, short_trend, medium_trend, long_trend, math_profit, etc.
      - Category groups: 'operational' ('go'), 'trend' ('gt'), 'financial' ('gf'), 'intelligence' ('gi')
      - Composite aliases: 'profit_guard' (bypasses GF1 + GF2)
      - Master bypass: 'all' or '*' (bypasses all guards for emergency exits)
    """
    bypassed: set[str] = set()
    if legacy_bypass_profit_guard:
        bypassed.add("profit_guard")
    if legacy_bypass_profit_reference:
        bypassed.add("math_profit")
    if legacy_bypass_quantity_policy:
        bypassed.add("quantity_policy")

    raw_tokens: list[str] = []
    if bypass_guards is not None:
        if bypass_guards is True or bypass_guards == "all" or bypass_guards == "*":
            raw_tokens.append("all")
        elif isinstance(bypass_guards, str):
            for part in bypass_guards.split(","):
                p = part.strip().lower()
                if p and p not in ("none", "off", "false", "no", "0"):
                    raw_tokens.append(p)
        elif isinstance(bypass_guards, (set, list, tuple)):
            for item in bypass_guards:
                p = str(item).strip().lower()
                if p and p not in ("none", "off", "false", "no", "0"):
                    raw_tokens.append(p)

    for token in raw_tokens:
        if token in _ALL_GUARD_ALIASES:
            bypassed.add(token)
        else:
            print(f"[GUARD] Warning: unrecognized bypass guard token {token!r} ignored; "
                  f"known guards are GO1, GT1..GT3, GF1..GF2, GI1..GI4")

    # Expand master bypass ('all' or '*')
    if "all" in bypassed or "*" in bypassed:
        bypassed.update({"all", "*", "profit_guard", "operational", "trend", "financial", "intelligence"})
        for g in GUARD_INSTANCES:
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    # Expand category groups ('operational', 'trend', 'financial', 'intelligence') and short prefixes
    category_shortcuts = {
        "go": "operational",
        "gt": "trend",
        "gf": "financial",
        "gi": "intelligence",
    }
    for sc, cat_name in category_shortcuts.items():
        if sc in bypassed:
            bypassed.add(cat_name)

    for cat_name, cat_guards in GUARD_CATEGORIES.items():
        if cat_name in bypassed:
            for g in cat_guards:
                bypassed.update(g.aliases)
                bypassed.add(g.canonical)
                bypassed.add(g.code.lower())

    # Expand legacy/composite profit_guard (GF1: math_profit + GF2: quantity_policy)
    if "profit_guard" in bypassed or "legacy_profit_guard" in bypassed:
        bypassed.update({"profit_guard", "legacy_profit_guard"})
        for code in ("GF1", "GF2"):
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
