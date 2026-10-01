#!/usr/bin/env python3
"""Tests for the funding scanner's ranking logic (pure, no API)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from funding_scan import rank  # noqa: E402


class TestRank(unittest.TestCase):
    def setUp(self):
        self.uni = [{"name": "AAA"}, {"name": "BBB"}, {"name": "CCC"}, {"name": "DDD"}]
        self.ctxs = [
            {"funding": "0.00005", "dayNtlVlm": "50000000", "markPx": "10", "openInterest": "1000"},  # 43.8%/year, liquid
            {"funding": "0.0001",  "dayNtlVlm": "1000000",  "markPx": "5",  "openInterest": "200"},   # 87%/year but ILLIQUID
            {"funding": "0.00002", "dayNtlVlm": "80000000", "markPx": "60", "openInterest": "500"},   # 17.5%/year, liquid
            {"funding": "-0.0001", "dayNtlVlm": "90000000", "markPx": "3",  "openInterest": "100"},   # funding NEGATIV
        ]

    def test_filtreaza_ilichid(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=10_000_000)
        coins = [x["coin"] for x in r]
        self.assertNotIn("BBB", coins, "the illiquid one (1M < 10M) must be filtered out despite its high funding")

    def test_filtreaza_funding_negativ(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=0)
        self.assertNotIn("DDD", [x["coin"] for x in r], "negative funding is not collected as a short")

    def test_claseaza_desc_pe_funding(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=10_000_000)
        self.assertEqual(r[0]["coin"], "AAA")          # 43.8% > 17.5%
        self.assertEqual(r[1]["coin"], "CCC")

    def test_apr_corect(self):
        r = rank(self.uni, self.ctxs, min_vol_usd=10_000_000)
        aaa = next(x for x in r if x["coin"] == "AAA")
        self.assertAlmostEqual(aaa["apr"], 0.00005 * 24 * 365 * 100, places=2)

    def test_top_limiteaza(self):
        self.assertLessEqual(len(rank(self.uni, self.ctxs, 0, top=1)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
