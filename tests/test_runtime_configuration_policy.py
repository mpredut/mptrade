from pathlib import Path
from types import SimpleNamespace
import subprocess
import re

from verify_tools.direct_order_api_inventory import inventory, scan_direct_calls
from botcore import resolve_cli_env_path
from binance_api import bapi_client
from unittest.mock import MagicMock, patch
import os
import sys
import tempfile
import textwrap
import unittest
from verify_tools.ownership_inventory import build_inventory, find_overlaps


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".conf", ".csv", ".env", ".html", ".ini", ".json", ".md", ".ps1",
    ".py", ".service", ".sh", ".toml", ".txt", ".yaml", ".yml",
}


def test_market_alerts_and_hyperliquid_provider_use_shared_env_stack():
    market = (ROOT / "market_monitor/price_watcher.py").read_text(encoding="utf-8")
    provider = (ROOT / "providers/hyperliquid_provider.py").read_text(encoding="utf-8")
    assert "load_env_stack(" in market
    assert "from dotenv import" not in market
    assert "load_env_stack(" in provider
    assert 'load_dotenv(os.path.join(_HL_DIR' not in provider


def test_operational_thresholds_have_one_versioned_source():
    config = (ROOT / "config.env").read_text(encoding="utf-8")
    keys = {
        "WATCHDOG_STALE_MINUTES",
        "WATCHDOG_COOLDOWN_MINUTES",
        "WATCHDOG_FAST_PRICE_MINUTES",
        "ANOMALY_WINDOW_FILES_MIN",
        "ANOMALY_COOLDOWN_MINUTES",
        "ANOMALY_MAX_READ_BYTES",
        "ANOMALY_THRESH_RATE_LIMIT",
        "ANOMALY_THRESH_AUTH",
        "ANOMALY_THRESH_BLIND",
        "ANOMALY_THRESH_TRACEBACK",
        "RISK_DD_ALERT_PCT",
        "RISK_DD_REALERT_HOURS",
        "RISK_DD_WORSEN_PCT",
        "MONITOR_PORT",
    }
    configured = {
        line.split("=", 1)[0].strip()
        for line in config.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert keys <= configured

    runtime_files = [
        path for path in [
            ROOT / "verify_tools/portfolio_snapshot.py",
            ROOT / "verify_tools/monitor_night.py",
        ] if path.exists()
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
    for key in keys:
        assert f'os.environ.get("{key}",' not in text


def test_specialized_atomic_writers_are_explicitly_bounded():
    # All bespoke atomic writers across the production fleet are fully migrated to state_io
    remaining = set()
    candidates = []
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=ROOT, text=True,
    ).splitlines()
    for relative in tracked:
        if (relative == "state_io.py"
                or relative.startswith(("tests/", "offline/", "archive/"))
                or "/test_" in relative):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        owns_temporary = re.search(
            r"\b(?:tmp|tmp_file|tmp_path|tmp_bin|temporary)\s*=.*(?:\.tmp|mkstemp)", text,
        )
        defines_local_atomic = re.search(
            r"^\s*def\s+(?:_?atomic_write(?:_json)?|_write_json_atomic)\b",
            text,
            re.MULTILINE,
        )
        if (owns_temporary and ("os.replace" in text or "mkstemp" in text)) or defines_local_atomic:
            candidates.append(relative)
    assert set(candidates) == remaining


def test_dead_trade_cache_manager_is_archived_not_packaged():
    assert (ROOT / "archive/legacy_archive.tar.gz").is_file()
    assert not (ROOT / "tradeCacheManager.py").exists()
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"tradeCacheManager"' not in packaging


def test_tracked_order_compatibility_shim_is_removed_after_consumer_migration():
    assert not (ROOT / "archive/providers_tracked_order.py").exists()
    assert not (ROOT / "providers/tracked_order.py").exists()
    active_imports = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if (set(relative.parts) & {"archive", "tests", ".venv", "myenv"}
                or any(part.startswith(".") for part in relative.parts)):
            continue
        text = path.read_text(encoding="utf-8")
        if ("providers.tracked_order" in text
                or "from providers import tracked_order" in text):
            active_imports.append(relative.as_posix())
    assert active_imports == []


def test_live_launcher_env_defaults_are_location_based(monkeypatch):
    monkeypatch.delenv("ENV_FILE", raising=False)
    assert resolve_cli_env_path("/tmp/venue_dir/launcher.py", []) == "/tmp/venue_dir/.env"
    for relative in (
        "kraken/kraken_bot.py",
        "hyperliquid/hl_bot.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert 'os.environ.get("ENV_FILE", ".env")' not in text
        assert (
            'os.path.dirname(__file__), ".env"' in text
            or "resolve_cli_env_path(__file__" in text
            or "load_cli_env_stack(__file__" in text
        )


def test_live_venue_credentials_use_complete_profile_resolver():
    files = (
        "kraken/kraken_bot.py",
        "kraken/kraken_xstock_watch.py",
        "kraken/trailing_stop.py",
        "providers/kraken_provider.py",
        "212trading/ipo.py",
        "212trading/t212_bot.py",
        "212trading/t212_status.py",
        "providers/t212_provider.py",
    )
    for relative in files:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "_credentials(" in text


def test_production_direct_order_boundaries_match_reviewed_allowlist():
    report = inventory()
    assert report["unapproved"] == []
    assert report["stale_allowlist"] == []


def test_new_direct_submit_is_detected(tmp_path: Path):
    path = tmp_path / "new_strategy.py"
    path.write_text(
        "def trade(client):\n    return client.order_market_buy(symbol='BTCUSDC', quantity=1)\n",
        encoding="utf-8",
    )
    calls = scan_direct_calls(tmp_path)
    assert {(call.path, call.function, call.method) for call in calls} == {
        ("new_strategy.py", "trade", "order_market_buy"),
    }


def _tracked_text_files():
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT,
    ).decode("utf-8")
    for relative in output.split("\0"):
        if not relative:
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            ".gitattributes", ".gitignore", "submit",
        }:
            yield path


def test_tracked_text_files_use_unix_line_endings():
    offenders = [
        str(path.relative_to(ROOT))
        for path in _tracked_text_files()
        if b"\r" in path.read_bytes()
    ]
    assert offenders == [], f"tracked text files contain CR/CRLF: {offenders}"




def _write_inv(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ownership_fixture() -> Path:
    root = Path(tempfile.mkdtemp())
    _write_inv(root, "procs.conf", """
dn_bot.py$|x|x|DN|||bot
kraken_bot.py|x|x|Kraken|||bot
kraken/trailing_stop.py|x|x|Trail|||bot
monitortrades.py|x||MT|||fleet
""")
    _write_inv(root, "config.txt", "trade_enabled=true\n")
    _write_inv(root, "config.env", "BINANCE_ACCOUNT_REF=binance-main\n")
    _write_inv(root, "kraken/config.env", """
KRAKEN_PAIR=HYPEUSD
STRAT_EXECUTE=true
KRAKEN_ACCOUNT_REF=kraken-main
""")
    _write_inv(root, "kraken/.env", "KRAKEN_LIVE_ORDERS=true\n")
    _write_inv(root, "kraken/trailing.conf", "KRAKEN_TRAILING_ENABLED=true\n")
    _write_inv(root, "hyperliquid/config.env", """
HL_COIN=HYPE
HL_SPOT_TOKEN=HYPE
STRAT_EXECUTE=true
HL_LIVE_ORDERS=true
""")
    _write_inv(root, "instruments.conf", """
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
    _write_inv(root, "binance_api/trailing.conf", "TRAILING_ENABLED=false\n")
    return root


class OwnershipInventoryTest(unittest.TestCase):
    def _make_fixture(self) -> Path:
        root = _ownership_fixture()
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        return root

    def test_two_primary_kraken_owners_are_warning_but_trailing_is_protective(self):
        root = self._make_fixture()
        owners = build_inventory(root)
        overlap = next(
            item for item in find_overlaps(owners)
            if item["venue"] == "kraken" and item["symbol"] == "HYPEUSD"
        )
        self.assertEqual(overlap["severity"], "warning")
        self.assertEqual(set(overlap["primary_owners"]), {"kraken-spot-dca", "monitortrades:KRAKEN_HYPE"})
        self.assertIn("kraken-trailing", overlap["owners"])

    def test_running_scope_omits_disabled_dn_and_non_running_trailing(self):
        root = self._make_fixture()
        owners = build_inventory(root, commands=["python3 kraken_bot.py", "python monitortrades.py"])
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
        root = self._make_fixture()
        owners = build_inventory(root, commands=["python3 hl_bot.py"])
        spot = next(owner for owner in owners if owner.owner_id == "hyperliquid-spot-dca")
        self.assertFalse(spot.configured)
        self.assertTrue(spot.live_enabled)
        self.assertTrue(spot.active(require_running=True))

    def test_explicit_monitortrades_account_removes_false_conflict(self):
        root = self._make_fixture()
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
        root = self._make_fixture()
        _write_inv(root, "procs.conf", """
tradeall.py|x||tradeall|||fleet
monitortrades.py|x||monitortrades|||fleet
""")
        _write_inv(root, "instruments.conf", """
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
        self.assertEqual(overlap["primary_coordination_domains"], ["binance-order-pipeline"])

    def test_config_txt_trade_enabled_false_overrides_config_env_true(self):
        root = self._make_fixture()
        _write_inv(root, "config.env", "BINANCE_ACCOUNT_REF=binance-main\nTRADE_ENABLED=true\n")
        _write_inv(root, "config.txt", "trade_enabled=false\n")
        owners = build_inventory(root)
        mt_owner = next(owner for owner in owners if owner.owner_id == "monitortrades:KRAKEN_HYPE")
        self.assertFalse(mt_owner.live_enabled)


class RuntimeImportSafetyTest(unittest.TestCase):
    def test_provider_layer_imports_without_binance_key_module(self):
        code = textwrap.dedent(
            """
            import importlib

            from providers.t212_provider import T212Provider
            from providers import market_api
            importlib.import_module("212trading.t212_client")

            if market_api._bapi is not None:
                raise AssertionError("Binance bapi a fost importat eager")
            T212Provider()
            if market_api.api.provider_by_name("t212") is None:
                raise AssertionError("T212 is missing from the registry after a direct import")

            from binance_api import bapi_ws
            if bapi_ws.api_key_ws:
                raise AssertionError("the test loaded a private key")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=dict(os.environ, BINANCE_AUTO_START_WEBSOCKETS="0", BINANCE_API_KEY_WS=""),
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cache_manager_import_is_network_and_worker_free(self):
        code = textwrap.dedent(
            """
            import threading
            import binance.client

            class ForbiddenClient:
                def __init__(self, *args, **kwargs):
                    raise AssertionError("Binance Client construit la import")

            binance.client.Client = ForbiddenClient
            import cacheManager
            from binance_api import bapi_trades

            forbidden = {
                "BinanceTimeResync", "CacheTradeManager",
                "CacheOrderManager",
                "CacheCurrentPriceManager", "NonBinanceTrendPoller",
                "InstantTrendFullEval", "InstantTrendFlush",
            }
            active = {thread.name for thread in threading.enumerate()}
            leaked = sorted(active & forbidden)
            if leaked:
                raise AssertionError(f"thread-uri pornite la import: {leaked}")
            """
        )
        env = dict(os.environ, BINANCE_AUTO_START_WEBSOCKETS="0",
                   BINANCE_API_KEY="", BINANCE_API_SECRET="", BINANCE_API_KEY_WS="")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_rest_credentials_fail_before_client_construction(self):
        code = textwrap.dedent('''
            from unittest.mock import patch
            from binance_api import bapi_client
            import credentials
            assert credentials.api_key_ws == "", "REST key must not substitute for the Ed25519 key"
            with patch.object(bapi_client, "Client") as client:
                try:
                    bapi_client.getClient()
                except ValueError as exc:
                    assert "BINANCE_API_SECRET" in str(exc)
                else:
                    raise AssertionError("Missing credentials were accepted")
                client.assert_not_called()
        ''')
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=dict(os.environ, BINANCE_API_KEY="test-only", BINANCE_API_SECRET="",
                     BINANCE_API_KEY_WS="", BINANCE_AUTO_START_WEBSOCKETS="0"),
            text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bapi_client_recvwindow_and_minus_1021_resync_retry(self):
        calls = []

        def _req_retry(method, uri, signed, force_params=False, **kwargs):
            data = kwargs.get("data")
            calls.append(dict(data) if isinstance(data, dict) else data)
            if len(calls) == 1:
                if isinstance(data, dict):
                    data["timestamp"] = 111
                    data["signature"] = "stale_sig"
                err = Exception("Timestamp outside recvWindow")
                err.code = -1021
                raise err
            return {"ok": True, "data": data}

        dummy = SimpleNamespace(
            session=MagicMock(),
            REQUEST_RECVWINDOW=None,
            timestamp_offset=0,
            get_server_time=lambda: {"serverTime": 1_700_000_005_000},
            _request=_req_retry,
        )
        bapi_client._install_recvwindow_retry(dummy)
        with patch.object(bapi_client, "_client", dummy):
            res = dummy._request("get", "https://api.binance.com/api/v3/account", True, data={"symbol": "BTCUSDC"})
        self.assertEqual(res["ok"], True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], {"symbol": "BTCUSDC"})
        self.assertEqual(bapi_client.REQUEST_RECVWINDOW_MS, 30000)

        def _make_html500():
            msg = "Invalid JSON error message from Binance: <html>\r\n<head><title>500 Internal Server Error</title></head>\r\n</html>"
            err = RuntimeError(f"APIError(code=0): {msg}")
            err.code = 0
            err.status_code = 500
            err.message = msg
            return err

        fail500 = SimpleNamespace(calls=0)
        fail500._request = lambda method, uri, signed, force_params=False, **kwargs: (
            setattr(fail500, "calls", fail500.calls + 1) or (_ for _ in ()).throw(_make_html500())
        )
        bapi_client._install_recvwindow_retry(fail500)
        with self.assertRaises(RuntimeError) as ctx:
            fail500._request("get", "https://api.binance.com/api/v3/account", True, data={"symbol": "BTCUSDC"})
        self.assertEqual(fail500.calls, 2)
        self.assertNotIn("\n", ctx.exception.message)
        self.assertNotIn("\r", ctx.exception.message)

    def test_get_client_disables_init_ping_and_installs_retries_before_sync(self):
        import credentials

        init_kwargs = {}
        events = []

        def _fake_sdk_client(api_key, api_secret, **kwargs):
            init_kwargs.update(kwargs)
            inst = SimpleNamespace(
                session=MagicMock(),
                REQUEST_RECVWINDOW=None,
                timestamp_offset=0,
                _request=lambda method, uri, signed, force_params=False, **kw: {"ok": True},
            )
            inst.get_server_time = lambda: (
                events.append(("get_server_time", getattr(inst, "_recvwindow_retry_installed", False)))
                or {"serverTime": 1_700_000_005_000}
            )
            return inst

        with (
            patch.object(bapi_client, "_client", None),
            patch.object(bapi_client, "Client", _fake_sdk_client),
            patch.object(bapi_client, "_start_periodic_resync"),
            patch.object(credentials, "api_key", "k"),
            patch.object(credentials, "api_secret", "s"),
        ):
            cl = bapi_client.getClient()

        self.assertIs(init_kwargs.get("ping"), False)
        self.assertEqual(events, [("get_server_time", True)])
        self.assertEqual(cl.REQUEST_RECVWINDOW, bapi_client.REQUEST_RECVWINDOW_MS)


def test_pyproject_python_version_and_active_packages_aligned():
    import tomllib
    import pytest
    import order_guard

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.12"
    assert pyproject["tool"]["black"]["target-version"] == ["py312"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py312"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.12"

    include_patterns = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])
    assert "shmem*" not in include_patterns
    expected_packages = {
        "212trading*",
        "binance_api*",
        "discovery*",
        "hyperliquid*",
        "intelligence*",
        "kraken*",
        "lock*",
        "market_monitor*",
        "notify_engine*",
        "orchestratorOS*",
        "orchestratorTrade*",
        "providers*",
        "strategies*",
        "verify_tools*",
        "visuals*",
    }
    assert expected_packages <= include_patterns

    base_margins = dict(order_guard._load_margins())
    with pytest.raises(ValueError, match="dynamic window progression"):
        order_guard._validate_margins_cfg(
            {**base_margins, "dynamic_window_favorable_h": 48.0, "dynamic_window_flat_h": 24.0},
            "order_guard.conf",
        )
    with pytest.raises(ValueError, match="parabolic bounds"):
        order_guard._validate_margins_cfg(
            {**base_margins, "parabolic_pullback_pct": 20.0, "parabolic_surge_pct": 15.0},
            "order_guard.conf",
        )
    with pytest.raises(ValueError, match="taker ratio bounds"):
        order_guard._validate_margins_cfg(
            {**base_margins, "min_taker_ratio": 0.95, "taker_downscale_threshold": 0.85},
            "order_guard.conf",
        )
    with pytest.raises(ValueError, match="orderbook imbalance bounds"):
        order_guard._validate_margins_cfg(
            {**base_margins, "min_buy_imbalance": 0.80, "max_sell_imbalance": 0.70},
            "order_guard.conf",
        )
    with pytest.raises(ValueError, match="funding rate bounds"):
        order_guard._validate_margins_cfg(
            {**base_margins, "funding_min_short_rate": 0.0001, "funding_max_long_rate": 0.0005},
            "order_guard.conf",
        )


