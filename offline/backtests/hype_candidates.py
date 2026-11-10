"""Pre-registered HYPE candidates, shared by the offline runners.

The list is deliberately small and fixed. It is not an optimisation grid and its
order is not a ranking. Every change stays off by default in live.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    overrides: dict


def default_candidates() -> list[Candidate]:
    """A small, interpretable neighbourhood around the live configuration."""
    return [
        Candidate("live", "the live configuration, unchanged", {}),
        Candidate("classic_tp", "a fixed TP; the trailing is off", {"tp_trend_hold": False}),
<<<<<<< Updated upstream
        Candidate("trail_2", "trailing 2%", {"tp_trail_adaptive": False, "tp_trail_pct": 2.0}),
        Candidate("trail_4", "trailing 4%", {"tp_trail_adaptive": False, "tp_trail_pct": 4.0}),
        Candidate("trail_5", "trailing 5%", {"tp_trail_adaptive": False, "tp_trail_pct": 5.0}),
        Candidate("tp_4", "TP threshold 4%", {"takeprofit_pct": 4.0}),
        Candidate("tp_6", "TP threshold 6%", {"takeprofit_pct": 6.0}),
=======
        Candidate("trail_2", "trailing 2%", {"tp_mdhtp_enabled": False, "tp_trail_adaptive": False, "tp_trail_pct": 2.0}),
        Candidate("trail_4", "trailing 4%", {"tp_mdhtp_enabled": False, "tp_trail_adaptive": False, "tp_trail_pct": 4.0}),
        Candidate("trail_5", "trailing 5%", {"tp_mdhtp_enabled": False, "tp_trail_adaptive": False, "tp_trail_pct": 5.0}),
        Candidate("tp_4", "prag TP 4%", {"takeprofit_pct": 4.0}),
        Candidate("tp_6", "prag TP 6%", {"takeprofit_pct": 6.0}),
>>>>>>> Stashed changes
        Candidate("dca_drop_1", "DCA on a 1% drop", {"dca_drop_pct": 1.0}),
        Candidate("dca_drop_1_5", "DCA on a 1.5% drop", {"dca_drop_pct": 1.5}),
        Candidate("dca_drop_2", "DCA on a 2% drop", {"dca_drop_pct": 2.0}),
        Candidate("reentry_1_5", "re-entry after -1.5%", {"reentry_drop_pct": 1.5}),
        Candidate("reentry_3", "re-entry after -3%", {"reentry_drop_pct": 3.0}),
        Candidate("sl_off", "the stop-loss is off", {"stop_loss_pct": 0.0}),
        Candidate("sl_10", "stop-loss 10%", {"stop_loss_pct": 10.0}),
        Candidate("sl_15", "stop-loss 15%", {"stop_loss_pct": 15.0}),
        Candidate("max_dca_6", "maximum 6 DCA", {"max_dca_buys": 6}),
        Candidate("lower_sizing", "entry 500 + DCA 250 USD", {"entry_amount": 500.0, "dca_amount": 250.0}),
        Candidate(
            "confirm_dca_1_5_trail_2",
<<<<<<< Updated upstream
            "confirmation: DCA 1.5% + trailing 2%",
            {"dca_drop_pct": 1.5, "tp_trail_adaptive": False, "tp_trail_pct": 2.0},
=======
            "confirmare: DCA 1,5% + trailing 2%",
            {"dca_drop_pct": 1.5, "tp_mdhtp_enabled": False, "tp_trail_adaptive": False, "tp_trail_pct": 2.0},
>>>>>>> Stashed changes
        ),
    ]


def _tp4(name: str) -> Candidate:
    return Candidate(name, "TP threshold 4%", {"takeprofit_pct": 4.0})


def _dca15(name: str) -> Candidate:
    return Candidate(name, "DCA on a 1.5% drop", {"dca_drop_pct": 1.5})


def _dca_vol_m1(name: str) -> Candidate:
    return Candidate(
        name,
        "DCA size scaled by volatility: k=-1, reference 2%",
        {
            "dca_vol_scale_k": -1.0,
            "dca_vol_ref": 2.0,
            "dca_vol_interval": 240,
        },
    )


def _adaptive_trail(name: str) -> Candidate:
    return Candidate(
        name, "adaptive trailing k=2, clamp 1.5-8%",
        {
            "tp_mdhtp_enabled": False,
            "tp_trail_adaptive": True,
            "tp_trail_k": 2.0,
            "tp_trail_min": 1.5,
            "tp_trail_max": 8.0,
            "tp_trail_vol_interval": 240,
        },
    )


def _dca_brake(name: str) -> Candidate:
    return Candidate(
        name, "block DCA in a confirmed downtrend",
        {
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.5,
            "trend_interval": 240,
        },
    )


def _overlay650t8() -> Candidate:
    return Candidate(
        "overlay650t8", "reduced overlay: top-up 650, trail 8%",
        {
            "trend_overlay": True,
            "trend_topup": 650.0,
            "trend_trail_pct": 8.0,
            "trend_interval": 240,
        },
    )


def hype_240_candidates() -> list[Candidate]:
    """The historical set of the walk-forward comparator; keeps the report names."""
    return [
        Candidate("live", "the live configuration, unchanged", {}),
        Candidate(
            "overlay_orig", "original overlay: top-up 2000, trail 5%",
            {"trend_overlay": True, "trend_topup": 2000.0,
             "trend_trail_pct": 5.0, "trend_interval": 240},
        ),
        _overlay650t8(),
        _adaptive_trail("A_adaptive_trail"),
        _dca_brake("B_dca_brake"),
        _tp4("tp_4"),
        _dca15("dca_drop_1_5"),
    ]


def financial_priority_candidates() -> list[Candidate]:
    """The candidates that go through the central/stress scenarios and the promotion gate."""
    return [
        Candidate("live", "the live configuration, unchanged", {}),
        Candidate(
            "reentry4",
            "re-entry only after a 4% pullback; HLC cross-venue shadow candidate",
            {"reentry_drop_pct": 4.0},
        ),
        _tp4("tp4"),
        _dca15("dca15"),
        Candidate(
            "dca_progressive025",
            "DCA 1.25% initially, then +0.25pp after each executed DCA",
            {"dca_spacing_growth_pct": 0.25},
        ),
        _dca_vol_m1("dca_vol_m1"),
        _adaptive_trail("A_trail"),
        _dca_brake("B_dcabrake"),
        _overlay650t8(),
        Candidate(
            "trail_profit_floor_sl18",
            "soft trailing at +1% gross minimum; hard MARKET stop at -18%",
            {"tp_trail_profit_floor_pct": 1.0, "stop_loss_pct": 18.0},
        ),
        Candidate(
            "trail_profit_floor_sl125",
            "DECOUPLED: a gross profit floor of +1%, but a hard stop at 12.5% (the baseline) "
            "— isolates profit-floor upside from the wide-stop drawdown penalty",
            {"tp_trail_profit_floor_pct": 1.0},
        ),
    ]
