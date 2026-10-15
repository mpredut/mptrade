"""guard_hierarchy.py — Canonical Object-Oriented Guard Hierarchy & SSOT Registry.

Implements the Separation of Concerns (SoC) for trading safety and intelligence guards:
1. OperationalGuards (G1, G2): Execution timing, rate-limiting, and micro-momentum deferral.
2. FinancialGuards (G3, G4): Mathematical price profitability and risk-weighted sizing.
3. IntelligenceGuards (G5..G8): Multi-pillar macro, micro, sentiment, and geopolitical threats.

Guards are opt-out: any guard not explicitly bypassed executes with its default behavior.
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
# Category 1: Operational & Timing Guards (G1, G2)
# ============================================================================

class OperationalGuard(BaseGuard):
    """Guards governing execution velocity, volume caps, and micro-timing."""
    category = "operational"


class DailyLimitGuard(OperationalGuard):
    code = "G1"
    canonical = "daily_limit"
    aliases = {"daily_limit", "daily_cap", "g1"}
    description = "Daily trade count cap and anti-spam temporal filter"


class ShortTrendGuard(OperationalGuard):
    code = "G2"
    canonical = "short_trend"
    aliases = {"short_trend", "trend", "trend_short", "g2"}
    description = "Short-term trend deferral via short_trend_manager"


# ============================================================================
# Category 2: Financial & Capital Protection Guards (G3, G4)
# ============================================================================

class FinancialGuard(BaseGuard):
    """Guards governing deterministic pricing profitability and capital exposure."""
    category = "financial"


class MathProfitGuard(FinancialGuard):
    code = "G3"
    canonical = "math_profit"
    aliases = {"math_profit", "math_reference", "profit_margin", "g3"}
    description = "Deterministic historical profit margin reference check"


class QuantityPolicyGuard(FinancialGuard):
    code = "G4"
    canonical = "quantity_policy"
    aliases = {"quantity_policy", "weight", "weight_sizing", "g4"}
    description = "Dynamic exposure and risk-weighted quantity sizing"


# ============================================================================
# Category 3: Market Intelligence & Threat Shield Guards (G5..G8)
# ============================================================================

class IntelligenceGuard(BaseGuard):
    """Guards evaluating market microstructure, momentum exhaustion, AI, and geopolitics."""
    category = "intelligence"


class Pillar1QuantitativeGuard(IntelligenceGuard):
    code = "G5"
    canonical = "p1_quantitative"
    aliases = {"p1_quantitative", "p1", "pillar1", "quantitative", "g5"}
    description = "Pillar 1: Parabolic Surge and Weibull trend exhaustion"


class Pillar2MicrostructureGuard(IntelligenceGuard):
    code = "G6"
    canonical = "p2_microstructure"
    aliases = {"p2_microstructure", "p2", "pillar2", "microstructure", "g6"}
    description = "Pillar 2: Orderbook Ask Wall, Funding crowding, and Whale flow"


class Pillar3LLMGuard(IntelligenceGuard):
    code = "G7"
    canonical = "p3_llm"
    aliases = {"p3_llm", "p3", "pillar3", "ai", "llm", "g7"}
    description = "Pillar 3: High-stake pre-trade LLM validation"


class Pillar4GeopoliticalGuard(IntelligenceGuard):
    code = "G8"
    canonical = "p4_geopolitical"
    aliases = {"p4_geopolitical", "p4", "pillar4", "geopolitical", "g8"}
    description = "Pillar 4: Geopolitical Black Swan shock shield"


# ============================================================================
# SSOT Registries & Lookup Mappings
# ============================================================================

GUARD_INSTANCES: tuple[BaseGuard, ...] = (
    DailyLimitGuard(),
    ShortTrendGuard(),
    MathProfitGuard(),
    QuantityPolicyGuard(),
    Pillar1QuantitativeGuard(),
    Pillar2MicrostructureGuard(),
    Pillar3LLMGuard(),
    Pillar4GeopoliticalGuard(),
)

GUARD_REGISTRY: dict[str, BaseGuard] = {g.code: g for g in GUARD_INSTANCES}

GUARD_CATEGORIES: dict[str, list[BaseGuard]] = {
    "operational": [g for g in GUARD_INSTANCES if isinstance(g, OperationalGuard)],
    "financial": [g for g in GUARD_INSTANCES if isinstance(g, FinancialGuard)],
    "intelligence": [g for g in GUARD_INSTANCES if isinstance(g, IntelligenceGuard)],
}

# Backward-compatible dictionary matching the previous GUARD_CATALOG schema
GUARD_CATALOG: dict[str, dict[str, Any]] = {
    g.code: {
        "canonical": g.canonical,
        "aliases": g.aliases,
        "description": g.description,
        "category": g.category,
    }
    for g in GUARD_INSTANCES
}

_ALL_GUARD_ALIASES: set[str] = set().union(*(g.aliases for g in GUARD_INSTANCES)) | {
    "all", "*", "profit_guard", "legacy_profit_guard", "cooldown",
    "operational", "financial", "intelligence",
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
      - Individual guard codes: 'G1'..'G8'
      - Canonical names or aliases: 'daily_limit', 'short_trend', 'math_profit', etc.
      - Category groups: 'operational', 'financial', 'intelligence'
      - Composite aliases: 'profit_guard' (bypasses G3 + G4)
      - Master bypass: 'all' or '*' (bypasses all G1..G8 for emergency exits)
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
                  f"known guards are G1..G8 ({', '.join(sorted(GUARD_REGISTRY.keys()))})")

    # Expand master bypass ('all' or '*')
    if "all" in bypassed or "*" in bypassed:
        bypassed.update({"all", "*", "profit_guard", "operational", "financial", "intelligence"})
        for g in GUARD_INSTANCES:
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    # Expand category groups ('operational', 'financial', 'intelligence')
    for cat_name, cat_guards in GUARD_CATEGORIES.items():
        if cat_name in bypassed:
            for g in cat_guards:
                bypassed.update(g.aliases)
                bypassed.add(g.canonical)
                bypassed.add(g.code.lower())

    # Expand legacy/composite profit_guard (G3: math_profit + G4: quantity_policy)
    if "profit_guard" in bypassed or "legacy_profit_guard" in bypassed:
        bypassed.update({"profit_guard", "legacy_profit_guard"})
        for code in ("G3", "G4"):
            g = GUARD_REGISTRY[code]
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    # Expand any individual G1..G8 token to include its canonical name and all alias variations
    for g in GUARD_INSTANCES:
        if bypassed.intersection(g.aliases) or g.code.lower() in bypassed:
            bypassed.update(g.aliases)
            bypassed.add(g.canonical)
            bypassed.add(g.code.lower())

    return bypassed
