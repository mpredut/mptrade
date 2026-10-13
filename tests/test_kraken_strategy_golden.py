"""A GOLDEN REGRESSION — it locks the base v2 behaviour (kraken/strategy.py) BEFORE
refactorul provider-agnostic (Calea B: unificare pe MarketDataProvider).

Runs the LIVE strategy through the faithful engine (replay.run_replay) over a
DETERMINISTIC slice of the frozen HYPE dataset and checks:
  1. the exact DECISION TRACE (every order placed: side/kind/price/vol/market) — stable hash.
  2. METRICILE finale (net/realized/fees/cycles/...) — la 8 zecimale.

After the refactor (rewiring strategy.py from KrakenClient to MarketDataProvider), THIS TEST
MUST PASS UNCHANGED — proof that the base v2 decisions are byte-identical. If it
fails, the refactor changed live behaviour (a regression) and must be stopped.

Valorile golden au fost capturate pe HEAD-ul dinainte de refactor (branch
refactor/provider-unify, Faza 0)."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kraken"))
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

# The venues have legacy modules with the same names (`strategy`, `replay`,
# `notify`). We load the Kraken graph under a unique name and restore the cache,
# so the result does not depend on the order in which pytest collects the files.
_COLLIDING = ("strategy", "replay", "market_data", "notify")
_PRELOADED = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
try:
    _SPEC = importlib.util.spec_from_file_location(
        "kraken_golden_replay_under_test", os.path.join(ROOT, "kraken", "replay.py")
    )
    rp = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = rp
    _SPEC.loader.exec_module(rp)
    strat = rp._strat
finally:
    for _name in _COLLIDING:
        sys.modules.pop(_name, None)
    sys.modules.update(_PRELOADED)

DATASET = os.path.join(ROOT, "offline", "research", "hype_dataset",
                       "HYPEUSDC_240m_hlspot.csv")
BARS = 800          # slice determinist
BUDGET = 3900.0

# --- GOLDEN (capturat pe branch refactor/provider-unify, Faza 0) -------------
GOLDEN_ORDERS = 14
GOLDEN_TRACE_HASH = "69fd0a5053d74c58"
GOLDEN_METRICS = {
    "realized": 1844.33939624,
    "net": 1817.57411476,
    "fees": 26.76528149,
    "total": 1817.57411476,
    "final_upnl": 0.0,
    "cycles": 4,
    "wins": 4,
    "maxdd": 297.20377694,
    "open_qty": 0.0,
    "fills": 13,
}


def _base_v2_params():
    return strat.StratParams(
        currency="USD", entry_amount=650.0, entry_discount_pct=0.8,
        dca_amount=325.0, dca_drop_pct=1.25, check_minutes=2.0,
        takeprofit_pct=5.0, max_budget=BUDGET, max_dca_buys=10,
        enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=12.5,
        adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.2,
        reentry_tolerance_pct=0.05, reentry_adaptive=False,
        reentry_sl_bounce_pct=1.5, tp_tranches=[], tp_trend_hold=True,
        tp_trail_pct=3.0, sl_mhdsl_enabled=False,
    )


def _load_ohlc(n):
    with open(DATASET, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
            for r in rows][:n]


class BaseV2GoldenTest(unittest.TestCase):
    """Amprenta de referinta pt refactorul provider-agnostic."""

    def _run_with_trace(self):
        ohlc = _load_ohlc(BARS)
        trace = []
        orig_place = strat.Strategy._place

        def _rec(self, side, vol, price, **kw):
            trace.append((side, kw.get("kind"),
                          None if price is None else round(price, 4),
                          round(vol, 6), bool(kw.get("market"))))
            return orig_place(self, side, vol, price, **kw)

        strat.Strategy._place = _rec
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = rp.run_replay(ohlc, _base_v2_params(),
                                        fee_pct=0.26, bar_minutes=240)
        finally:
            strat.Strategy._place = orig_place
        return trace, metrics

    def test_decision_trace_is_byte_identical(self):
        trace, _ = self._run_with_trace()
        self.assertEqual(len(trace), GOLDEN_ORDERS,
                         "the number of orders placed changed -> a behaviour regression")
        digest = hashlib.sha256(repr(trace).encode()).hexdigest()[:16]
        self.assertEqual(digest, GOLDEN_TRACE_HASH,
                         "urma exacta de decizii s-a schimbat -> refactorul a alterat base v2")

    def test_final_metrics_match_golden(self):
        _, m = self._run_with_trace()
        for key, expected in GOLDEN_METRICS.items():
            got = m.get(key)
            if isinstance(expected, float):
                self.assertAlmostEqual(got, expected, places=6, msg=f"metrica '{key}' difera")
            else:
                self.assertEqual(got, expected, f"metrica '{key}' difera")


if __name__ == "__main__":
    unittest.main(verbosity=2)
