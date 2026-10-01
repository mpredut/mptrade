from pathlib import Path
import tempfile
import unittest

from verify_tools.ownership_inventory import build_inventory, find_overlaps


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture() -> Path:
    root = Path(tempfile.mkdtemp())
    _write(root, "procs.conf", """
dn_bot.py$|x|x|DN|||bot
kraken_bot.py|x|x|Kraken|||bot
kraken/trailing_stop.py|x|x|Trail|||bot
monitortrades.py|x||MT|||fleet
""")
    _write(root, "config.txt", "trade_enabled=true\n")
    _write(root, "config.env", "BINANCE_ACCOUNT_REF=binance-main\n")
    _write(root, "kraken/config.env", """
KRAKEN_PAIR=HYPEUSD
STRAT_EXECUTE=true
KRAKEN_ACCOUNT_REF=kraken-main
""")
    _write(root, "kraken/.env", "KRAKEN_LIVE_ORDERS=true\n")
    _write(root, "kraken/trailing.conf", "KRAKEN_TRAILING_ENABLED=true\n")
    _write(root, "hyperliquid/config.env", """
HL_COIN=HYPE
HL_SPOT_TOKEN=HYPE
STRAT_EXECUTE=true
HL_LIVE_ORDERS=true
""")
    _write(root, "instruments.conf", """
[KRAKEN_HYPE]
provider=kraken
symbol=HYPEUSD
base=HYPE
quote=USD
enabled=yes
mt.gain=9

[HYPERLIQUID_HYPE]
provider=hyperliquid
symbol=HYPEUSDC
base=HYPE
quote=USDC
enabled=no
mt.gain=9
""")
    _write(root, "symbols.py", "symbols=[]\ntaosymbol='TAOUSDC'\n")
    _write(root, "binance_api/trailing.conf", "TRAILING_ENABLED=false\n")
    return root


class OwnershipInventoryTest(unittest.TestCase):
    def test_two_primary_kraken_owners_are_warning_but_trailing_is_protective(self):
        root = _fixture()
        owners = build_inventory(root)

        overlap = next(
            item for item in find_overlaps(owners)
            if item["venue"] == "kraken" and item["symbol"] == "HYPEUSD"
        )
        self.assertEqual(overlap["severity"], "warning")
        self.assertEqual(
            set(overlap["primary_owners"]),
            {"kraken-spot-dca", "monitortrades:KRAKEN_HYPE"},
        )
        self.assertIn("kraken-trailing", overlap["owners"])

    def test_running_scope_omits_disabled_dn_and_non_running_trailing(self):
        root = _fixture()
        commands = ["python3 kraken_bot.py", "python monitortrades.py"]
        owners = build_inventory(root, commands=commands)

        dn = next(owner for owner in owners if owner.owner_id == "hyperliquid-dn")
        self.assertTrue(dn.configured)
        self.assertTrue(dn.live_enabled)
        self.assertFalse(dn.active(require_running=True))
        overlap = next(
            item for item in find_overlaps(owners, require_running=True)
            if item["venue"] == "kraken"
        )
        self.assertNotIn("kraken-trailing", overlap["owners"])

    def test_running_scope_detects_owner_started_outside_manifest(self):
        root = _fixture()
        owners = build_inventory(root, commands=["python3 hl_bot.py"])

        spot = next(
            owner for owner in owners
            if owner.owner_id == "hyperliquid-spot-dca"
        )
        self.assertFalse(spot.configured)
        self.assertTrue(spot.live_enabled)
        self.assertTrue(spot.active(require_running=True))

    def test_explicit_monitortrades_account_removes_false_conflict(self):
        root = _fixture()
        instruments = root / "instruments.conf"
        instruments.write_text(
            instruments.read_text(encoding="utf-8").replace(
                "enabled=yes\nmt.gain=9",
                "enabled=yes\nownership.account_ref=kraken-spare\nmt.gain=9",
                1,
            ),
            encoding="utf-8",
        )
        owners = build_inventory(root)

        warnings = [
            item for item in find_overlaps(owners)
            if item["severity"] == "warning" and item["venue"] == "kraken"
        ]
        self.assertEqual(warnings, [])

    def test_same_coordination_domain_is_informational_not_warning(self):
        root = _fixture()
        _write(root, "procs.conf", """
tradeall.py|x||tradeall|||fleet
monitortrades.py|x||monitortrades|||fleet
""")
        _write(root, "symbols.py", "symbols=['BTCUSDC']\ntaosymbol='TAOUSDC'\n")
        _write(root, "instruments.conf", """
[BINANCE_BTC]
provider=binance
symbol=BTCUSDC
base=BTC
quote=USDC
enabled=yes
mt.gain=5
""")

        overlaps = find_overlaps(build_inventory(root))

        overlap = next(item for item in overlaps if item["venue"] == "binance")
        self.assertEqual(overlap["severity"], "info")
        self.assertEqual(
            overlap["primary_coordination_domains"],
            ["binance-order-pipeline"],
        )


if __name__ == "__main__":
    unittest.main()
