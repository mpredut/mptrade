"""Pure walk-forward metrics for composite market-regime decisions."""

from __future__ import annotations

import math


def evaluate_forward_classifications(samples):
    """Score decisions against already-observed forward returns.

    Each sample is ``(decision, forward_return)``. Unknown/non-actionable decisions
    contribute to coverage but never count as directional predictions.
    """
    total = directional = correct = conflicts = 0
    signed_returns = []
    by_pattern = {}
    for decision, raw_return in samples:
        total += 1
        value = float(raw_return)
        if not math.isfinite(value):
            raise ValueError("forward returns must be finite")
        if decision.conflict:
            conflicts += 1
        bucket = by_pattern.setdefault(
            decision.pattern, {"count": 0, "directional": 0, "correct": 0})
        bucket["count"] += 1
        if not decision.actionable or decision.regime not in {"bull", "bear"}:
            continue
        directional += 1
        bucket["directional"] += 1
        predicted_sign = 1.0 if decision.regime == "bull" else -1.0
        signed_returns.append(predicted_sign * value)
        if predicted_sign * value > 0:
            correct += 1
            bucket["correct"] += 1
    return {
        "samples": total,
        "directional": directional,
        "coverage": directional / total if total else 0.0,
        "directional_accuracy": correct / directional if directional else 0.0,
        "mean_signed_forward_return": (
            sum(signed_returns) / len(signed_returns) if signed_returns else 0.0),
        "conflict_rate": conflicts / total if total else 0.0,
        "patterns": by_pattern,
    }
