"""The root config.env must not define keys that a venue loads into os.environ.

log.py and accepted_order_persistence.py load the root config.env at import time, before a
venue bot calls load_env_stack() on its own config.env, and load_dotenv never overwrites.
A key present in both therefore takes the ROOT value in the venue process, silently: from
19 to 24 Sep 2026 the root's legacy STRAT_EXECUTE=false / STRAT_TAKEPROFIT_PCT=0.02 turned
the Kraken and Hyperliquid bots into paper scalpers.

Trading 212 profiles are read with parse_dotenv() into per-profile dictionaries, not into
os.environ, so they are not covered here.
"""
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from botcore import load_env_stack, parse_dotenv  # noqa: E402

VENUES = ("kraken", "hyperliquid")


class RootConfigIsolationTest(unittest.TestCase):
    def test_root_config_defines_no_venue_key(self):
        root_keys = set(parse_dotenv(os.path.join(ROOT, "config.env")))
        for venue in VENUES:
            venue_keys = set(parse_dotenv(os.path.join(ROOT, venue, "config.env")))
            with self.subTest(venue=venue):
                self.assertEqual(sorted(root_keys & venue_keys), [])

    def test_venue_strategy_survives_root_config_loaded_first(self):
        # Reproduce the bot import order: root config first, then the venue stack.
        for venue in VENUES:
            venue_cfg = parse_dotenv(os.path.join(ROOT, venue, "config.env"))
            with self.subTest(venue=venue), patch.dict(os.environ, {}, clear=True):
                from botcore import load_dotenv
                load_dotenv(os.path.join(ROOT, "config.env"))
                load_env_stack(os.path.join(ROOT, venue, ".env.absent-for-test"))
                for key in ("STRAT_EXECUTE", "STRAT_TAKEPROFIT_PCT", "STRAT_DCA_DROP_PCT"):
                    if key in venue_cfg:
                        self.assertEqual(os.environ.get(key), venue_cfg[key], key)


if __name__ == "__main__":
    unittest.main()
