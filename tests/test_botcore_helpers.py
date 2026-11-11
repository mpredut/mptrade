import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import unittest
from unittest.mock import MagicMock, patch

import pytest

from notify_engine import alertnotifiers
import botcore
from botcore import load_env_stack, normalize_scalar_order_id, parse_dotenv
from cachepaths import (
    cache_path,
    get_cachedb_dir,
    organize_cachedb_subdomains,
    resolve_cache_subdomain,
    resolve_cachedb_filepath,
)
from credentials import (
    CredentialConfigurationError,
    kraken_credentials,
    t212_credentials,
)
import utils
from providers.base import mount_connect_retry
from strategies.state_store import JsonStateStore
import requests
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "offline" / "runners"


def _response(status=200, body=b"ok"):
    response = MagicMock()
    response.status = status
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


class SharedHttpHelpersTest(unittest.TestCase):
    def test_json_and_form_requests_share_transport_without_changing_wire_format(self):
        with patch.object(
            botcore.urllib.request, "urlopen", return_value=_response(201, b"done")
        ) as urlopen:
            result = botcore.http_post_json(
                "https://example.test/orders", {"qty": 2}, {"Authorization": "x"}
            )
        request = urlopen.call_args.args[0]
        self.assertEqual(result, (201, b"done"))
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {"qty": 2})
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("Authorization"), "x")

        with patch.object(
            botcore.urllib.request, "urlopen", return_value=_response()
        ) as urlopen:
            botcore.http_post_form("https://example.test/orders", {"pair": "HYPE/USD"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.data, b"pair=HYPE%2FUSD")
        self.assertEqual(
            request.get_header("Content-type"),
            "application/x-www-form-urlencoded",
        )

    def test_http_and_transport_errors_keep_previous_fail_closed_contract(self):
        error = urllib.error.HTTPError(
            "https://example.test", 429, "rate limit", {}, io.BytesIO(b"slow down")
        )
        with patch.object(botcore.urllib.request, "urlopen", side_effect=error):
            self.assertEqual(botcore.http_get("https://example.test"),
                             (429, b"slow down"))

        with patch.object(
            botcore.urllib.request, "urlopen", side_effect=TimeoutError("timeout")
        ), patch.object(botcore, "log"):
            self.assertEqual(botcore.http_request("DELETE", "https://example.test"),
                             (0, b""))

    def test_json_and_form_payloads_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            botcore.http_request(
                "POST", "https://example.test", payload={"x": 1}, form={"x": 1}
            )


class DotenvHelpersTest(unittest.TestCase):
    def test_load_and_parse_share_syntax_but_keep_their_override_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# comment\n"
                "export FIRST=one # inline\n"
                "QUOTED='two # preserved'\n"
                "DUP=first\n"
                "DUP=last\n",
                encoding="utf-8",
            )

            parsed = botcore.parse_dotenv(str(path))
            self.assertEqual(parsed["FIRST"], "one")
            self.assertEqual(parsed["QUOTED"], "two # preserved")
            self.assertEqual(parsed["DUP"], "last")

            with patch.dict(os.environ, {"FIRST": "real"}, clear=True), patch.object(
                botcore, "log"
            ):
                botcore.load_dotenv(str(path))
                self.assertEqual(os.environ["FIRST"], "real")
                self.assertEqual(os.environ["QUOTED"], "two # preserved")
                self.assertEqual(os.environ["DUP"], "first")

    def test_required_settings_reject_missing_empty_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "REQUIRED_X"):
                botcore.required_env("REQUIRED_X")
        with patch.dict(os.environ, {"REQUIRED_X": " # comment"}, clear=True):
            with self.assertRaisesRegex(ValueError, "REQUIRED_X"):
                botcore.required_env("REQUIRED_X")
        with patch.dict(os.environ, {"REQUIRED_X": "infinity"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                botcore.required_float_env("REQUIRED_X")
        with patch.dict(os.environ, {"REQUIRED_X": "1.5"}, clear=True):
            with self.assertRaisesRegex(ValueError, "integer"):
                botcore.required_int_env("REQUIRED_X")
        with patch.dict(os.environ, {"REQUIRED_X": "yes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "boolean"):
                botcore.required_bool_env("REQUIRED_X")

    def test_required_settings_parse_explicit_values_and_inline_comments(self):
        env = {
            "TEXT": "value # explanation",
            "FLOAT": "1.25 # seconds",
            "INT": "4 # attempts",
            "BOOL": "false # kill switch",
        }
        self.assertEqual(botcore.required_env("TEXT", env), "value")
        self.assertEqual(botcore.required_float_env("FLOAT", env), 1.25)
        self.assertEqual(botcore.required_int_env("INT", env), 4)
        self.assertFalse(botcore.required_bool_env("BOOL", env))
        self.assertEqual(botcore.float_env("FLOAT", env), 1.25)
        self.assertEqual(botcore.int_env("INT", env), 4)
        self.assertIsNone(botcore.int_env("FLOAT", env))
        self.assertIsNone(botcore.int_env("MISSING", env))


class NumericCoercionHelpersTest(unittest.TestCase):
    def test_safe_float_and_pos_and_nonneg_float(self):
        self.assertEqual(botcore.safe_float("12.5", 0.0), 12.5)
        self.assertEqual(botcore.safe_float(None, 0.0), 0.0)
        self.assertEqual(botcore.safe_float(True, 0.0), 0.0)
        self.assertEqual(botcore.safe_float(False, 0.0), 0.0)
        self.assertEqual(botcore.safe_float(float("nan"), -1.0), -1.0)
        self.assertEqual(botcore.safe_float(float("inf"), -1.0), -1.0)
        self.assertIsNone(botcore.safe_float("bad", None))

        self.assertEqual(botcore.safe_pos_float("10.5", 0.0), 10.5)
        self.assertEqual(botcore.safe_pos_float("0.0", 5.0), 5.0)
        self.assertEqual(botcore.safe_pos_float("-2.5", 5.0), 5.0)
        self.assertIsNone(botcore.safe_pos_float("bad", None))

        self.assertEqual(botcore.safe_nonneg_float("0.0", -1.0), 0.0)
        self.assertEqual(botcore.safe_nonneg_float("15.2", -1.0), 15.2)
        self.assertEqual(botcore.safe_nonneg_float("-0.01", 1.0), 1.0)
        self.assertIsNone(botcore.safe_nonneg_float("invalid", None))

    def test_safe_int_and_pos_int(self):
        self.assertEqual(botcore.safe_int("42", 0), 42)
        self.assertEqual(botcore.safe_int(42.9, 0), 42)
        self.assertEqual(botcore.safe_int(True, 0), 0)
        self.assertEqual(botcore.safe_int(False, 0), 0)
        self.assertEqual(botcore.safe_int(None, 0), 0)
        self.assertEqual(botcore.safe_int(float("nan"), -1), -1)
        self.assertIsNone(botcore.safe_int("not_a_number", None))

        self.assertEqual(botcore.safe_pos_int("5", 1), 5)
        self.assertEqual(botcore.safe_pos_int("0", 1), 1)
        self.assertEqual(botcore.safe_pos_int("-3", 1), 1)
        self.assertEqual(botcore.safe_pos_int(True, 1), 1)
        self.assertIsNone(botcore.safe_pos_int("bad", None))

        large_id = (1 << 53) + 7
        self.assertEqual(botcore.safe_int(large_id, 0), large_id)
        self.assertEqual(botcore.safe_int(str(large_id), 0), large_id)
        self.assertEqual(botcore.safe_pos_int(str(large_id), None), large_id)
        self.assertEqual(botcore.exact_int(large_id, 0), large_id)
        self.assertEqual(botcore.exact_int(str(large_id), 0), large_id)
        self.assertEqual(botcore.exact_int("5.0", 0), 5)
        self.assertIsNone(botcore.exact_int("5.5", None))
        self.assertIsNone(botcore.exact_int(5.5, None))
        self.assertIsNone(botcore.exact_pos_int(0, None))
        self.assertEqual(botcore.exact_nonneg_int(0, None), 0)
        from cachepaths import normalize_venue_name
        self.assertEqual(normalize_venue_name(SimpleNamespace(name="Kraken_Spot")), "kraken")
        self.assertIsNone(normalize_venue_name(SimpleNamespace(name=True), default=None))
        from order_retry import exact_int
        from state_io import exact_nonneg_int
        self.assertEqual(normalize_scalar_order_id(large_id), large_id)
        self.assertEqual(exact_int(large_id), large_id)
        self.assertEqual(exact_nonneg_int(large_id, default=None), large_id)

    def test_clamp_parse_ratio_validate_env_and_heartbeat_helpers(self):
        self.assertEqual(botcore.clamp_float(1.5, 0.0, 1.0, default=0.0), 1.0)
        self.assertEqual(botcore.clamp_float(-0.2, 0.0, 1.0, default=0.0), 0.0)
        self.assertEqual(botcore.clamp_float("0.42", 0.0, 1.0, default=0.0), 0.42)
        self.assertEqual(botcore.clamp_float("bad", 0.0, 1.0, default=0.25), 0.25)

        self.assertEqual(botcore.parse_ratio_or_pct("2.5"), 0.025)
        self.assertEqual(botcore.parse_ratio_or_pct("0.015"), 0.015)

        botcore.validate_required_env_keys(("A", "B"), env={"A": "1", "B": "2"}, label="TestMod")
        with self.assertRaisesRegex(ValueError, "Missing or empty mandatory TestMod settings: B"):
            botcore.validate_required_env_keys(("A", "B"), env={"A": "1", "B": "   "}, label="TestMod")

        with tempfile.TemporaryDirectory() as directory:
            hb_path = str(Path(directory) / "sub" / "hb.heartbeat")
            ts = botcore.touch_heartbeat_file(hb_path, last_ts=0.0, force=True, now=123456.78)
            self.assertEqual(ts, 123456.78)
            self.assertTrue(Path(hb_path).exists())

    def test_unified_config_env_and_ini_parsers(self):
        self.assertEqual(botcore.parse_dotenv_line("export FOO = 'bar' # comment"), ("FOO", "bar"))
        self.assertIsNone(botcore.parse_dotenv_line("# comment only", strict=True))
        with self.assertRaises(ValueError):
            botcore.parse_dotenv_line("NO_EQUALS_SIGN", strict=True)
        with self.assertRaises(ValueError):
            botcore.parse_dotenv_line("=missing_key", strict=True)

        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / "sample.env"
            env_path.write_text(
                "export Alpha=true # inline\nBeta=12.5\nGamma='hello'\n",
                encoding="utf-8",
            )
            parsed_upper = botcore.parse_dotenv(env_path, upper_keys=True)
            self.assertEqual(parsed_upper, {"ALPHA": "true", "BETA": "12.5", "GAMMA": "hello"})

            parsed_lower_coerced = botcore.parse_dotenv(
                env_path, lower_keys=True, coerce_bools=True, coerce_floats=True
            )
            self.assertEqual(
                parsed_lower_coerced,
                {"alpha": True, "beta": 12.5, "gamma": "hello"},
            )

            ini_path = Path(directory) / "sample.conf"
            ini_path.write_text("[sec]\nkey = val1 # inline\n", encoding="utf-8")
            cp1 = botcore.load_ini_config(ini_path)
            self.assertEqual(cp1.get("sec", "key"), "val1")
            cp2 = botcore.load_ini_config(ini_path)
            self.assertIs(cp1, cp2)

            ini_path.write_text("[sec]\nkey = val_updated_longer\n", encoding="utf-8")
            cp3 = botcore.load_ini_config(ini_path)
            self.assertEqual(cp3.get("sec", "key"), "val_updated_longer")
            self.assertIsNot(cp1, cp3)


class BoundNotifierTest(unittest.TestCase):
    def test_explicit_symbol_then_environment_priority_then_fallback(self):
        bound = alertnotifiers.bind_notify(
            ("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO"
        )
        with patch.object(alertnotifiers, "notify") as shared, patch.dict(
            os.environ, {"SYMBOL_LABEL": "LABEL", "KRAKEN_PAIR": "PAIR"}, clear=False
        ):
            bound("title", "body", "source")
            self.assertEqual(shared.call_args.args[3], "LABEL")
            self.assertIsNone(shared.call_args.kwargs["email"])
            bound("title", "body", "source", symbol="EXPLICIT")
            self.assertEqual(shared.call_args.args[3], "EXPLICIT")

        with patch.object(alertnotifiers, "notify") as shared, patch.dict(
            os.environ, {}, clear=True
        ):
            bound("title", "body", "source")
            self.assertEqual(shared.call_args.args[3], "CRYPTO")


class NotificationIsolationTest(unittest.TestCase):
    def test_external_notification_kill_switch_blocks_all_delivery(self):
        with patch.dict(os.environ, {"DISABLE_EXTERNAL_NOTIFICATIONS": "true"}, clear=False), \
             patch.object(alertnotifiers.AlertNotifier, "send_phone_webhook_batch") as ntfy, \
             patch.object(alertnotifiers.AlertNotifier, "send_email_batch") as email:
            alertnotifiers.notify(
                "STOP-LOSS test", "paper event", "pytest", "TSTX",
                desktop=True, email=True,
            )
        ntfy.assert_not_called()
        email.assert_not_called()


class TestCachepaths(unittest.TestCase):
    def test_mptrade_cachedb_dir_override(self):
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": "/custom/mptrade/cache"}, clear=False):
            self.assertEqual(get_cachedb_dir(), "/custom/mptrade/cache")
            path = cache_path("test.json")
            self.assertEqual(path.replace("\\", "/"), "/custom/mptrade/cache/test.json")

    def test_binance_cache_dir_fallback(self):
        env = dict(os.environ)
        env.pop("MPTRADE_CACHEDB_DIR", None)
        env["BINANCE_CACHE_DIR"] = "/custom/binance/cache"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_cachedb_dir(), "/custom/binance/cache")
            path = cache_path("test.json")
            self.assertEqual(path.replace("\\", "/"), "/custom/binance/cache/test.json")

    def test_default_dir_ends_with_cachedb(self):
        env = dict(os.environ)
        env.pop("MPTRADE_CACHEDB_DIR", None)
        env.pop("BINANCE_CACHE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            d = get_cachedb_dir()
            self.assertTrue(d.replace("\\", "/").endswith("cachedb"))
            self.assertTrue(cache_path("test.json").replace("\\", "/").endswith("cachedb/test.json"))

    def test_preserves_absolute_paths(self):
        abs_path = os.path.abspath("already/absolute/file.json")
        self.assertEqual(cache_path(abs_path), abs_path)

    def test_empty_or_none(self):
        self.assertEqual(cache_path(""), "")
        self.assertIsNone(cache_path(None))

    def test_utils_cache_path_integration(self):
        with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": "/tmp/custom_cache"}, clear=False):
            p = utils.cache_path("my_cache.json")
            self.assertEqual(p.replace("\\", "/"), "/tmp/custom_cache/my_cache.json")

    def test_resolve_cache_subdomain_covers_all_thematic_domains_and_sidecars(self):
        cases = {
            "cache_current_price.json": "prices",
            "cache_24_price_BTCUSDC.json.meta": "prices",
            "cache_365_price_SOLUSDC.json.previous": "prices",
            "cache_sparse_price_TAOUSD.json": "prices",
            "cache_365_price_BTCUSDC.npz": "prices",
            "cache_short_trend.json": "trends",
            "cache_long_trend.json.lock": "trends",
            "cache_kalman_trend.json": "trends",
            "24trend": "trends",
            "orderbook_depth_BTCUSDC.json": "telemetry",
            "derivatives_telemetry_BTCUSDC.json": "telemetry",
            "whale_snapshot_ETHUSDC.json": "telemetry",
            "fear_greed_index.json": "macro",
            "market_breadth_metrics.json": "macro",
            "news_feed_snapshot.json": "macro",
            "macro_lexicon_collect.json": "macro",
            "sentiment_eval.json": "macro",
            "geopolitical_eval.json": "macro",
            "cache_order.json": "orders",
            "cache_order_binance.json": "orders",
            "cache_trade_binance.jsonl": "orders",
            "cache_orders_kraken.json": "orders",
            "cache_trades_hyperliquid.json": "orders",
            "cache_asset_value_t212.json": "orders",
            "binance_account_cache_health.json": "orders",
            "order_retry_queue.json": "orders",
            "assetguardian_state_BTCUSDC.json": "state",
            "mt_state.json": "state",
            "rtrade_pairs_state.json": "state",
            "trailing_state.json": "state",
            "smart_stop_loss_state.json": "state",
            "autonomous_trade_intents_state.json": "state",
            "rtrade_BTCUSDC.heartbeat": "system/heartbeat",
            "auto_technical_repair.heartbeat": "system/heartbeat",
            "auto_technical_repair.heartbeat.previous": "system/heartbeat",
            "auto_repair_eval.json": "system",
            "fleet_health_status.json": "system",
            "dev_regression_status.json": "system",
            "llm_topic_threads.json": "system",
            "notify_queue.json": "system",
        }
        for filename, expected_domain in cases.items():
            self.assertEqual(
                resolve_cache_subdomain(filename),
                expected_domain,
                f"Expected {filename} -> {expected_domain}",
            )
        self.assertIsNone(resolve_cache_subdomain("unclassified_custom_file.json"))

    def test_organize_cachedb_subdomains_moves_files_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cache_current_price.json").write_text('{"BTC": 100}', encoding="utf-8")
            (root / "cache_current_price.json.meta").write_text("{}", encoding="utf-8")
            (root / "cache_short_trend.json").write_text("{}", encoding="utf-8")
            (root / "fear_greed_index.json").write_text("{}", encoding="utf-8")
            (root / "rtrade_BTCUSDC.heartbeat").write_text("123", encoding="utf-8")
            (root / "system").mkdir(exist_ok=True)
            (root / "system" / "macro_analyzer.heartbeat").write_text("456", encoding="utf-8")
            (root / "orders").mkdir(exist_ok=True)
            (root / "orders" / "cache_order.json").write_text("{}", encoding="utf-8")
            (root / "orders" / "cache_trade.jsonl").write_text("", encoding="utf-8")
            stale_dir = root / ".cache_24_price_BTCUSDC.json.previous.abcd123"
            stale_dir.mkdir()
            (stale_dir / "staged.tmp").write_text("old", encoding="utf-8")

            moved = organize_cachedb_subdomains(tmpdir, remove_stale_staging=True)
            self.assertEqual(
                sorted(moved["prices"]),
                ["cache_current_price.json", "cache_current_price.json.meta"],
            )
            self.assertEqual(moved["trends"], ["cache_short_trend.json"])
            self.assertEqual(moved["macro"], ["fear_greed_index.json"])
            self.assertEqual(
                sorted(moved["orders"]),
                ["cache_order_binance.json", "cache_trade_binance.jsonl"],
            )
            self.assertEqual(
                sorted(moved["system/heartbeat"]),
                ["macro_analyzer.heartbeat", "rtrade_BTCUSDC.heartbeat"],
            )
            self.assertFalse(stale_dir.exists())
            self.assertTrue((root / "prices" / "cache_current_price.json").exists())
            self.assertTrue((root / "trends" / "cache_short_trend.json").exists())
            self.assertTrue((root / "macro" / "fear_greed_index.json").exists())
            self.assertTrue((root / "orders" / "cache_order_binance.json").exists())
            self.assertTrue((root / "orders" / "cache_trade_binance.jsonl").exists())
            self.assertFalse((root / "orders" / "cache_order.json").exists())
            self.assertFalse((root / "orders" / "cache_trade.jsonl").exists())
            self.assertTrue((root / "system" / "heartbeat" / "rtrade_BTCUSDC.heartbeat").exists())
            self.assertTrue((root / "system" / "heartbeat" / "macro_analyzer.heartbeat").exists())
            self.assertFalse((root / "system" / "macro_analyzer.heartbeat").exists())

            resolved = resolve_cachedb_filepath(os.path.join(tmpdir, "cache_current_price.json"))
            self.assertEqual(
                Path(resolved),
                root / "prices" / "cache_current_price.json",
            )
            with patch.dict(os.environ, {"MPTRADE_CACHEDB_DIR": tmpdir}, clear=False):
                self.assertEqual(
                    Path(cache_path("orderbook_depth_BTCUSDC.json")),
                    root / "telemetry" / "orderbook_depth_BTCUSDC.json",
                )


def test_env_stack_preserves_process_secret_and_versioned_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / "profile.env"
    config_file = tmp_path / "config.env"
    env_file.write_text(
        "STACK_SHARED=secret-file\nSTACK_SECRET_ONLY=secret\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "STACK_SHARED=versioned\nSTACK_CONFIG_ONLY=config\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STACK_SHARED", "process")
    monkeypatch.delenv("STACK_SECRET_ONLY", raising=False)
    monkeypatch.delenv("STACK_CONFIG_ONLY", raising=False)

    load_env_stack(str(env_file))

    assert os.environ["STACK_SHARED"] == "process"
    assert os.environ["STACK_SECRET_ONLY"] == "secret"
    assert os.environ["STACK_CONFIG_ONLY"] == "config"


def test_env_stack_resolves_config_next_to_selected_env_file(tmp_path, monkeypatch):
    nested = tmp_path / "venue"
    nested.mkdir()
    env_file = nested / ".env"
    env_file.write_text("", encoding="utf-8")
    (nested / "runtime.env").write_text("STACK_ADJACENT=yes\n", encoding="utf-8")
    monkeypatch.delenv("STACK_ADJACENT", raising=False)

    load_env_stack(str(env_file), "runtime.env")

    assert os.environ["STACK_ADJACENT"] == "yes"


def test_venue_env_stack_overrides_root_config_env(tmp_path, monkeypatch):
    root_cfg = tmp_path / "root_config.env"
    root_cfg.write_text("VENUE_FALLBACK=root_val\nVENUE_ROOT_ONLY=root_only\n", encoding="utf-8")
    monkeypatch.setattr(botcore, "_ROOT_CONFIG_PATH", str(root_cfg.resolve()))
    monkeypatch.delenv("VENUE_FALLBACK", raising=False)
    monkeypatch.delenv("VENUE_ROOT_ONLY", raising=False)
    monkeypatch.delenv("VENUE_CFG_ONLY", raising=False)

    botcore.load_dotenv(str(root_cfg))
    assert os.environ["VENUE_FALLBACK"] == "root_val"
    assert os.environ["VENUE_ROOT_ONLY"] == "root_only"

    venue_dir = tmp_path / "venue"
    venue_dir.mkdir()
    venue_cfg = venue_dir / "config.env"
    venue_cfg.write_text("VENUE_FALLBACK=venue_val\nVENUE_CFG_ONLY=venue_only\n", encoding="utf-8")
    venue_env = venue_dir / ".env"
    venue_env.write_text("", encoding="utf-8")

    load_env_stack(str(venue_env))

    assert os.environ["VENUE_FALLBACK"] == "venue_val"
    assert os.environ["VENUE_CFG_ONLY"] == "venue_only"
    assert os.environ["VENUE_ROOT_ONLY"] == "root_only"


def test_root_config_defines_no_venue_key():
    root_dir = str(ROOT)
    root_keys = set(botcore.parse_dotenv(os.path.join(root_dir, "config.env")))
    for venue in ("kraken", "hyperliquid"):
        venue_keys = set(botcore.parse_dotenv(os.path.join(root_dir, venue, "config.env")))
        assert sorted(root_keys & venue_keys) == [], f"Conflict for {venue}"


def test_venue_strategy_survives_root_config_loaded_first():
    root_dir = str(ROOT)
    for venue in ("kraken", "hyperliquid"):
        venue_cfg = botcore.parse_dotenv(os.path.join(root_dir, venue, "config.env"))
        with patch.dict(os.environ, {}, clear=True):
            botcore.load_dotenv(os.path.join(root_dir, "config.env"))
            botcore.load_env_stack(os.path.join(root_dir, venue, ".env.absent-for-test"))
            for key in ("STRAT_EXECUTE", "STRAT_TAKEPROFIT_PCT", "STRAT_DCA_DROP_PCT"):
                if key in venue_cfg:
                    assert os.environ.get(key) == venue_cfg[key], f"{key} in {venue}"


def test_versioned_dev_profile_has_every_required_field():
    profile = parse_dotenv(str(RUNNERS / "dev_backtest.env"))
    required = {
        "DEV_HOST", "DEV_PORT", "DEV_USER", "DEV_PATH",
        "DEV_CODE_BRANCH", "BACKTEST_PROPOSALS_BRANCH",
    }
    assert required <= profile.keys()
    assert all(profile[key].strip() for key in required)


def test_shell_profile_loader_exports_the_versioned_values():
    env = dict(os.environ, RUNNER_DIR=str(RUNNERS))
    result = subprocess.run(
        ["bash", "-c", "source offline/runners/load_dev_backtest_env.sh; "
         "printf '%s|%s|%s' \"$DEV_HOST\" \"$DEV_CODE_BRANCH\" "
         "\"$BACKTEST_PROPOSALS_BRANCH\""],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    profile = parse_dotenv(str(RUNNERS / "dev_backtest.env"))
    assert result.stdout == "|".join([
        profile["DEV_HOST"], profile["DEV_CODE_BRANCH"],
        profile["BACKTEST_PROPOSALS_BRANCH"],
    ])


def test_kraken_cache_uses_complete_dedicated_pair():
    pair = kraken_credentials("cache", values={
        "KRAKEN_API_KEY_CACHE": "cache-key",
        "KRAKEN_API_SECRET_CACHE": "cache-secret",
        "KRAKEN_API_KEY_BOT": "bot-key",
        "KRAKEN_API_SECRET_BOT": "bot-secret",
    })
    assert (pair.key, pair.secret, pair.profile) == (
        "cache-key", "cache-secret", "Kraken CACHE",
    )


def test_kraken_cache_fallback_is_an_explicit_complete_pair():
    pair = kraken_credentials("cache", values={
        "KRAKEN_API_KEY_BOT": "bot-key",
        "KRAKEN_API_SECRET_BOT": "bot-secret",
    })
    assert pair.profile == "Kraken BOT fallback"


def test_partial_primary_profile_does_not_hide_behind_fallback():
    with pytest.raises(CredentialConfigurationError, match="CACHE"):
        kraken_credentials("cache", values={
            "KRAKEN_API_KEY_CACHE": "partial",
            "KRAKEN_API_KEY_BOT": "bot-key",
            "KRAKEN_API_SECRET_BOT": "bot-secret",
        })


def test_fleet_never_mixes_key_and_secret_from_different_profiles():
    with pytest.raises(CredentialConfigurationError, match="primary"):
        kraken_credentials("fleet", values={
            "KRAKEN_API_KEY": "primary-key",
            "KRAKEN_API_SECRET_SPARE": "spare-secret",
            "KRAKEN_API_KEY_BOT": "bot-key",
            "KRAKEN_API_SECRET_BOT": "bot-secret",
        })


def test_t212_requires_both_halves():
    with pytest.raises(CredentialConfigurationError, match="T212"):
        t212_credentials(values={"T212_API_KEY": "key-only"})


def test_monitororder_ttl_and_ordered_id_set():
    import monitororder
    # Custom TTL parsing
    ttl = monitororder.resolve_effective_order_ttl("123", "client_456", custom_ttl_map={"123": "300"})
    assert ttl == 300.0
    ttl_invalid = monitororder.resolve_effective_order_ttl("123", "client_456", custom_ttl_map={"123": -10})
    assert ttl_invalid == monitororder.DEFAULT_ORDER_TTL_SEC

    # OrderedIdSet trim
    s = monitororder.OrderedIdSet()
    for i in range(10):
        s.add(str(i))
    assert len(s) == 10
    s.trim(5)
    assert len(s) == 5
    assert list(s) == ["5", "6", "7", "8", "9"]


def test_kraken_cache_touch_ws_fetchtimes():
    from cacheaccount import CacheTradeManager
    from providers.kraken_provider import KrakenProvider
    pairs = KrakenProvider.resolve_cache_daemon_config()["pairs"]
    cache = {"items": {"TAOUSDC": []}, "fetchtime": {}}
    CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(cache, now_ms=1700000000, symbols=pairs, create_missing=True)
    assert cache["fetchtime"]["TAOUSDC"] == 1700000000
    CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(cache, now_ms="invalid", symbols=pairs, create_missing=True)
    assert cache["fetchtime"]["TAOUSDC"] > 0


def test_autonomous_intent_gateway_extract_price():
    from botcore import extract_current_price_item
    assert extract_current_price_item(100.5) == 100.5
    assert extract_current_price_item({"price": "101.2"}) == 101.2
    assert extract_current_price_item({"c": 102.3}) == 102.3
    assert extract_current_price_item([[1, 103.4]]) == 103.4
    assert extract_current_price_item([104.5]) == 104.5
    assert extract_current_price_item(None) is None
    assert extract_current_price_item(True) is None
    assert extract_current_price_item(-5.0) is None


def test_price_analysis_parse_strict_int():
    from priceAnalysis import require_int
    assert require_int(5, error_msg="not integral") == 5
    assert require_int(5.0, error_msg="not integral") == 5
    assert require_int("5.0", error_msg="not integral") == 5
    with pytest.raises(ValueError, match="not integral"):
        require_int(3.5, error_msg="not integral")
    with pytest.raises(ValueError, match="not integral"):
        require_int(None, error_msg="not integral")
    with pytest.raises(ValueError, match="not integral"):
        require_int(True, error_msg="not integral")


def test_execution_audit_json_safe_value():
    from decimal import Decimal
    from providers.execution_audit import _json_safe_value, _parse_cache_max
    assert _json_safe_value(Decimal("123.45")) == 123.45
    assert _json_safe_value(Decimal("nan")) is None
    assert _json_safe_value(float("inf")) is None
    assert _json_safe_value({"a": Decimal("10.0"), "b": [1, 2.5]}) == {"a": 10.0, "b": [1, 2.5]}
    assert _parse_cache_max() >= 100


def test_iteration186_load_local_module_should_wait_short_trend_and_strip_owner_tags(tmp_path):
    from botcore import load_local_module
    from notify_engine.alertnotifiers import strip_owner_tags
    from pricewindow import should_wait_short_trend
    from trailing_core import vwap_or_fallback

    # 1. strip_owner_tags cleans bracketed and parenthesized bot attribution tags
    assert strip_owner_tags("[TRADEALL] BUY BTCUSDC") == " BUY BTCUSDC"
    assert strip_owner_tags("(MANUAL) SELL ETHUSDC") == " SELL ETHUSDC"
    assert strip_owner_tags("[ASSETGUARDIAN] REENTRY") == " REENTRY"
    assert strip_owner_tags(None) == ""
    assert strip_owner_tags(True) == ""

    # 2. vwap_or_fallback computes weighted average or safely falls back
    assert vwap_or_fallback(100.0, 2.0, 45.0) == pytest.approx(50.0)
    assert vwap_or_fallback(0.0, 2.0, 45.0) == pytest.approx(45.0)
    assert vwap_or_fallback(100.0, 0.0, 45.0) == pytest.approx(45.0)
    assert vwap_or_fallback(None, None, 45.0) == pytest.approx(45.0)
    assert vwap_or_fallback("nan", 2.0, "bad") == pytest.approx(0.0)

    # 3. should_wait_short_trend evaluates short trend deferral
    class DummyShortTM:
        def __init__(self, wait_res):
            self.wait_res = wait_res
        def should_wait(self, side, symbol, **kw):
            return self.wait_res

    assert should_wait_short_trend(DummyShortTM(True), "SELL", "BTCUSDC") is True
    assert should_wait_short_trend(DummyShortTM(False), "SELL", "BTCUSDC") is False

    # 4. load_local_module supports custom module_key and register=False, and ensure_local_module_priority evicts foreign modules
    from botcore import ensure_local_module_priority
    mod_dir = tmp_path / "venue_mod"
    mod_dir.mkdir()
    (mod_dir / "sample_local.py").write_text("VALUE = 42\n", encoding="utf-8")
    mod = load_local_module(str(mod_dir), "sample_local", module_key="custom_sample_local", register=False)
    assert mod.VALUE == 42
    assert "custom_sample_local" not in sys.modules

    foreign_mod = type(sys)("sample_local")
    foreign_mod.__file__ = str(tmp_path / "other_dir" / "sample_local.py")
    sys.modules["sample_local"] = foreign_mod
    orig_sys_path = list(sys.path)
    try:
        ensure_local_module_priority(str(mod_dir), "sample_local")
        assert "sample_local" not in sys.modules
        assert sys.path[0] == os.path.abspath(str(mod_dir))
    finally:
        sys.path[:] = orig_sys_path


def test_shared_intelligence_and_agent_primitives_promoted_to_core():
    from cachepaths import clean_symbol_key

    assert botcore.obj_float({"x": "12.5"}, "x") == pytest.approx(12.5)
    assert botcore.clean_json_value({"a": float("nan"), "b": [1.5, float("inf")]}) == {"a": None, "b": [1.5, None]}
    assert botcore.safe_thresh("75.4", 50) == 75
    assert botcore.safe_thresh("150", 50) == 50
    assert botcore.clean_scale("65%", default=1.0, allow_percent=True) == pytest.approx(0.65)
    assert botcore.clean_scale(50, default=1.0, allow_percent=True) == pytest.approx(0.5)
    assert botcore.resolve_now_ts(1700000000.0) == pytest.approx(1700000000.0)
    assert botcore.is_cache_fresh(100.0, 110.0, 30.0) is True
    assert botcore.is_cache_fresh(100.0, 110.0, 30.0, force_refresh=True) is False
    assert botcore.is_interval_due(150.0, 100.0, 30.0) is True
    assert botcore.is_interval_due(110.0, 100.0, 30.0) is False
    cd = {}
    assert botcore.check_cooldown_gate(cd, "k", 100.0, cooldown_sec=60.0) is True
    assert botcore.check_cooldown_gate(cd, "k", 120.0, cooldown_sec=60.0) is False
    assert botcore.split_csv(" a, b , a\nc ", upper=True) == ("A", "B", "C")
    assert botcore.parse_symbol_list("btcusdc, ethusdc") == ["BTCUSDC", "ETHUSDC"]
    assert clean_symbol_key("tao/usd-perp_1") == "TAOUSDPERP1"
    assert botcore.resolve_order_notional(None, 250.0, 3.0) == pytest.approx(750.0)
    assert botcore.resolve_order_notional(800.0, 250.0, 3.0) == pytest.approx(800.0)


def test_unified_heartbeat_api_and_cached_http_collector(tmp_path):
    from intelligence.macro.daemon_ssot import CachedHttpJsonCollector, run_scheduled_step

    hb_path = tmp_path / "rtrade_BTCUSDC.heartbeat"
    ts = botcore.touch_heartbeat_file(
        str(hb_path),
        last_ts=0.0,
        force=True,
        now=1700000100.0,
        status={"ts": 1700000100.0, "cycle": 1},
        symbols=["BTCUSDC"],
        intervals={"loop": 5},
    )
    assert ts == pytest.approx(1700000100.0)
    payload = json.loads(hb_path.read_text(encoding="utf-8"))
    assert payload["status"] == "alive"
    assert payload["ts"] == pytest.approx(1700000100.0)
    assert payload["last_run"] == {"ts": 1700000100.0, "cycle": 1}
    assert payload["symbols"] == ["BTCUSDC"]
    assert payload["intervals"] == {"loop": 5}
    assert isinstance(payload["pid"], int)

    mtime = os.path.getmtime(str(hb_path))
    is_fresh, age_sec, max_stale_duration, stale_file = botcore.evaluate_heartbeat_files_freshness(
        [str(hb_path)], stale_threshold_s=60.0, now_ts=mtime + 20.0
    )
    assert is_fresh is True
    assert age_sec == pytest.approx(20.0)
    assert max_stale_duration == pytest.approx(0.0)
    assert stale_file == ""

    class DummyRunner:
        last_step_ts = 0.0
        calls = 0
        def _step(self, x):
            self.calls += x
            return {"count": self.calls}

    runner = DummyRunner()
    res = {}
    ran = run_scheduled_step(
        runner,
        current_ts=100.0,
        ts_attr="last_step_ts",
        interval_sec=30.0,
        fn=runner._step,
        fn_args=(2,),
        results=res,
        result_key="step",
    )
    assert ran is True
    assert runner.last_step_ts == pytest.approx(100.0)
    assert runner.calls == 2
    assert res["step"] is True
    assert run_scheduled_step(
        runner,
        current_ts=110.0,
        ts_attr="last_step_ts",
        interval_sec=30.0,
        fn=runner._step,
        fn_args=(2,),
        results=res,
        result_key="step",
    ) is False

    class SampleCollector(CachedHttpJsonCollector):
        def _parse_fetched_payload(self, data, now):
            return {"value": data.get("v", 0), "ts": now} if isinstance(data, dict) else None

    import intelligence.macro_sentiment_ssot as mss
    collector = SampleCollector(60.0, 60.0, "https://example.test/api", lambda: "https://example.test/api")
    with patch.object(mss, "fetch_public_json", return_value={"v": 42}) as mock_fetch:
        first = collector.fetch(now_ts=1000.0)
        second = collector.fetch(now_ts=1020.0)
        assert first == {"value": 42, "ts": 1000.0}
        assert second == {"value": 42, "ts": 1000.0}
        assert mock_fetch.call_count == 1


def test_load_local_module_thread_safe_concurrent_execution(tmp_path):
    import threading

    mod_path = tmp_path / "slow_venue_client.py"
    mod_path.write_text(
        "import time\n"
        "time.sleep(0.05)\n"
        "class VenueClient:\n"
        "    READY = True\n",
        encoding="utf-8",
    )
    sys.modules.pop("slow_venue_client", None)
    barrier = threading.Barrier(6)
    results = []
    errors = []

    def _worker():
        try:
            barrier.wait(timeout=2.0)
            mod = botcore.load_local_module(str(tmp_path), "slow_venue_client")
            results.append(mod.VenueClient.READY)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)
    sys.modules.pop("slow_venue_client", None)
    assert errors == []
    assert results == [True] * 6


def test_canonical_package1_unified_primitives():
    import types
    from intelligence.macro_sentiment_ssot import lookup_symbol_cache_items

    # 1. strict_ascii_int
    assert botcore.strict_ascii_int(42, min_value=1) == 42
    assert botcore.strict_ascii_int(" 123 ", min_value=1) == 123
    assert botcore.strict_ascii_int("123.0", allow_float_str=False) is None
    assert botcore.strict_ascii_int("123.0", allow_float_str=True) == 123
    assert botcore.strict_ascii_int("١٢٣", default=None) is None  # non-ASCII digits rejected
    assert botcore.strict_ascii_int(True, default=None) is None
    assert botcore.strict_ascii_int(0, min_value=1) is None

    # 2. normalize_epoch_sec & resolve_now_ts
    assert botcore.normalize_epoch_sec(1_700_000_000_000) == pytest.approx(1_700_000_000.0)
    assert botcore.normalize_epoch_sec(1_700_000_000.5) == pytest.approx(1_700_000_000.5)
    assert botcore.normalize_epoch_sec(0, allow_zero=False) is None
    assert botcore.normalize_epoch_sec(0, allow_zero=True) == 0.0
    assert botcore.normalize_epoch_sec("2026-01-01T00:00:00Z", allow_iso=True) == pytest.approx(1767225600.0)
    assert botcore.resolve_now_ts(1_700_000_000_000) == pytest.approx(1_700_000_000.0)

    # 3. resolve_patched_attr
    default_fn = lambda: "default"
    patched_fn = lambda: "patched"
    dummy_mod = types.ModuleType("_dummy_patch_target")
    dummy_mod.fn = patched_fn
    sys.modules["_dummy_patch_target"] = dummy_mod
    try:
        resolved = botcore.resolve_patched_attr(
            "fn",
            default_fn,
            local_val=default_fn,
            modules=("_dummy_patch_target",),
            require_callable=True,
        )
        assert resolved is patched_fn
    finally:
        sys.modules.pop("_dummy_patch_target", None)

    # 4. lookup_symbol_cache_items superset modes
    wrapped = {"items": {"BTCUSDT": [[1000, 50000.0]], "TAOUSD": {"median_d": 3.5}}}
    assert lookup_symbol_cache_items(wrapped, "BTCUSDC", require_list=True, unwrap_items=True) == [[1000, 50000.0]]
    assert lookup_symbol_cache_items(wrapped, "TAOUSDC", require_dict=True, unwrap_items=True) == {"median_d": 3.5}
    single_wrapped = {"items": {"HYPEUSD": [[1000, 25.0]]}}
    assert lookup_symbol_cache_items(
        single_wrapped,
        "HYPE",
        allow_base_fallback=False,
        single_item_quote_fallback=True,
        require_list=True,
        unwrap_items=True,
    ) == [[1000, 25.0]]

    # 5. Wave 2 primitives: require_*, clean_json_value, normalize_trend_direction, sanitizers, lru, order_id, cache root
    from collections import OrderedDict
    from botcore import clean_order_id_str
    from cachepaths import cachedb_root_for_file
    from providers.order_ohlc_utils import parse_ohlc_interval_minutes

    assert botcore.require_finite_float("3.5", "x") == 3.5
    assert botcore.require_pos_float("2.0", "x", min_value=1.5) == 2.0
    assert botcore.require_int("7", "n", min_value=1) == 7
    with pytest.raises(ValueError, match="x must be finite and positive"):
        botcore.require_pos_float(0.0, "x")
    with pytest.raises(ValueError, match="invalid count"):
        botcore.require_int(0, "count", min_value=1)

    cleaned_no_str = botcore.clean_json_value({1: [float("nan"), (2.5,)]}, stringify_keys=False)
    assert cleaned_no_str == {1: [None, [2.5]]}

    assert botcore.normalize_trend_direction("BULLISH", preserve_bull_bear=False) == "up"
    assert botcore.normalize_trend_direction("BULLISH", preserve_bull_bear=True) == "bull"
    assert botcore.normalize_trend_direction("up", preserve_bull_bear=True) == "up"
    assert botcore.normalize_trend_direction("short", allow_long_short=True) == "down"
    assert botcore.normalize_trend_direction("short", default="neutral", allow_long_short=False) == "neutral"

    assert botcore.sanitize_pipe_field("a|b\r\nc") == "a/b  c"
    assert botcore.sanitize_identifier("..my/id$1..", strip_chars="._") == "my_id_1"
    assert botcore.sanitize_identifier("a.b", allow_dots_hyphens=False) == "a_b"
    assert botcore.normalize_class_name(OrderedDict) == "OrderedDict"
    assert botcore.normalize_class_name("  MyCache ") == "MyCache"

    lru = OrderedDict()
    botcore.lru_dict_put(lru, "a", 1, 2)
    botcore.lru_dict_put(lru, "b", 2, 2)
    botcore.lru_dict_put(lru, "a", 10, 2)
    botcore.lru_dict_put(lru, "c", 3, 2)
    assert list(lru.items()) == [("a", 10), ("c", 3)]

    assert clean_order_id_str(None) == ""
    assert clean_order_id_str(" 12345.0 ") == "12345"
    assert normalize_scalar_order_id(42.0) == 42
    assert normalize_scalar_order_id("ABC-1") == "ABC-1"
    assert parse_ohlc_interval_minutes("5m", str_to_min={}, allow_unit_multiplier=True, require_unit_suffix=True) == 5
    assert cachedb_root_for_file("/tmp/cachedb/trends/cache_trend.json") == "/tmp/cachedb"
    assert cachedb_root_for_file("/tmp/cachedb/system/heartbeat/hb.json") == "/tmp/cachedb"

    import importlib
    saved_botcore_dict = dict(botcore.__dict__)
    try:
        pre_reload_resolve = botcore.resolve_patched_attr
        pre_unset = botcore._UNSET
        importlib.reload(botcore)
        assert botcore._UNSET is pre_unset
        assert pre_reload_resolve("NONEXISTENT_ATTR", 42, modules=("botcore",)) == 42
    finally:
        botcore.__dict__.update(saved_botcore_dict)


class HttpResilienceTest(unittest.TestCase):
    def _retry(self, **kw):

        session = requests.Session()
        self.assertTrue(mount_connect_retry(session, **kw))
        return session.get_adapter("https://api.example.com").max_retries

    def test_connect_only_is_the_default_and_never_retries_read_or_status(self):
        retry = self._retry()
        self.assertEqual(retry.connect, 3)
        self.assertEqual(retry.read, 0)
        self.assertEqual(retry.status, 0)

    def test_both_schemes_share_the_mounted_adapter(self):

        session = requests.Session()
        mount_connect_retry(session)
        self.assertIs(session.get_adapter("https://x").max_retries,
                      session.get_adapter("http://x").max_retries)

    def test_idempotent_profile_adds_read_and_status_but_excludes_post(self):
        retry = self._retry(attempts=3, backoff=1.0, read=3,
                            status_forcelist=[429, 502, 503, 504], raise_on_status=True)
        self.assertEqual(retry.connect, 3)
        self.assertEqual(retry.read, 3)
        self.assertEqual(sorted(retry.status_forcelist), [429, 502, 503, 504])
        self.assertTrue(retry.raise_on_status)
        allowed = (getattr(retry, "allowed_methods", None)
                   or getattr(retry, "method_whitelist", None))
        self.assertIsNotNone(allowed, "urllib3 Retry should expose an idempotent set")
        self.assertNotIn("POST", {str(m).upper() for m in allowed})

    def test_best_effort_returns_false_without_raising(self):

        class _BadSession:
            def mount(self, *_a, **_k):
                raise RuntimeError("cannot mount")

        self.assertFalse(mount_connect_retry(_BadSession()))


if __name__ == "__main__":
    unittest.main()

# ============================================================================
# Domain Regressions & Input Hygiene (Targeted Validation)
# ============================================================================


def test_botcore_diff_percent_are_close_reject_bool_and_nan_iter47_s1():
    """[ITER47 #1] botcore: diff_percent, are_close reject bool and NaN."""
    import math
    # 1. botcore: diff_percent, are_close reject bool and NaN
    assert math.isnan(botcore.diff_percent(1.0, True))
    assert math.isnan(botcore.diff_percent(float("nan"), 100.0))
    assert botcore.are_close(1.0, True, 0.05) is False
    assert botcore.are_close(100.0, 100.0, float("nan")) is False
    assert botcore.are_close(100.0, 100.0, True) is False
    assert botcore.are_close(100.0, 100.04, 0.05) is True

def test_credentials_resolve_credential_pair_rejects_boolean_values_instead_of_stringifying_iter63_s2():
    """[ITER63 #2] credentials.resolve_credential_pair rejects boolean values instead of stringifying True -> 'True'."""
    from credentials import CredentialProfileMissingError, resolve_credential_pair
    # 2. credentials.resolve_credential_pair rejects boolean values instead of stringifying True -> "True"
    with pytest.raises(CredentialProfileMissingError):
        resolve_credential_pair(
            (("Test", "K", "S"),),
            values={"K": True, "S": True},
        )

def test_safe_float_is_us_rth_open_handle_overflowerror_booleans_and_out_of_iter65_s2():
    """[ITER65 #2] safe_float & is_us_rth_open handle OverflowError, booleans, and out-of-range timestamps."""
    from botcore import is_us_rth_open, safe_float
    # 2. safe_float & is_us_rth_open handle OverflowError, booleans, and out-of-range timestamps
    assert safe_float(10**400, 12.5) == 12.5
    assert isinstance(is_us_rth_open(True), bool)
    assert isinstance(is_us_rth_open(float("inf")), bool)
    assert isinstance(is_us_rth_open(1e20), bool)

def test_utils_gaussian_weights_from_idx_handles_overflowerror_and_clamps_huge_t_iter71_s5():
    """[ITER71 #5] utils.gaussian_weights_from_idx handles OverflowError and clamps huge T."""
    # 5. utils.gaussian_weights_from_idx handles OverflowError and clamps huge T
    t_arr, w_arr = utils.gaussian_weights_from_idx(10**400, 0)
    assert len(t_arr) == 0 and len(w_arr) == 0

def test_config__parse_bool_flag_handles_nan_inf_and_unrecognized_strings_iter81_s3():
    """[ITER81 #3] config._parse_bool_flag handles NaN/Inf and unrecognized strings by falling back to default."""
    import botcore as config
    # 3. config._parse_bool_flag handles NaN/Inf and unrecognized strings by falling back to default
    assert config._parse_bool_flag(float("nan"), default=False) is False
    assert config._parse_bool_flag(float("nan"), default=True) is True
    assert config._parse_bool_flag("unknown_token", default=True) is True
    assert config._parse_bool_flag("unknown_token", default=False) is False

def test_jsonstatestore_rejects_non_dict_loaded_payloads_even_if_iter84_s2(monkeypatch, tmp_path):
    """[ITER84 #2] JsonStateStore rejects non-dict loaded payloads even if load_json_state returns a list of pairs."""
    from strategies.state_store import StatePersistenceError
    # 2. JsonStateStore rejects non-dict loaded payloads even if load_json_state returns a list of pairs
    monkeypatch.setattr("strategies.state_store.load_json_state", lambda *a, **k: [["cycle", 99]])
    store = JsonStateStore(
        str(tmp_path / "state.json"),
        lambda: {"cycle": 1, "qty": 0.0},
        label="test",
        logger=lambda _: None,
        fail_closed=True,
    )
    with pytest.raises(StatePersistenceError):
        store.load()

def test_http_resilience_mount_connect_retry_rejects_boolean_negative_and_non_finite_iter84_s4():
    """[ITER84 #4] http_resilience.mount_connect_retry rejects boolean, negative, and non-finite retry parameters."""
    from providers import base as http_resilience
    # 4. http_resilience.mount_connect_retry rejects boolean, negative, and non-finite retry parameters
    sess = requests.Session()
    assert http_resilience.mount_connect_retry(sess, attempts=False) is False
    assert http_resilience.mount_connect_retry(sess, attempts=-1) is False
    assert http_resilience.mount_connect_retry(sess, backoff=float("nan")) is False
    assert http_resilience.mount_connect_retry(sess, attempts=2, backoff=0.2) is True

def test_botcore__dotenv_pairs_strips_trailing_inline_comments_after_quoted_iter89_s2(tmp_path):
    """[ITER89 #2] botcore._dotenv_pairs strips trailing inline comments after quoted values while preserving # inside quotes."""
    # 2. botcore._dotenv_pairs strips trailing inline comments after quoted values while preserving # inside quotes
    env_file = tmp_path / "test.env"
    env_file.write_text(
        'A="val_a" # trailing comment\n'
        "B='val#b' # another comment\n"
        'C="val#c"\n',
        encoding="utf-8",
    )
    parsed = botcore.parse_dotenv(str(env_file))
    assert parsed == {"A": "val_a", "B": "val#b", "C": "val#c"}

def test_credentials_kraken_credentials_normalizes_role_case_whitespace_and_rejects_iter89_s3():
    """[ITER89 #3] credentials.kraken_credentials normalizes role case/whitespace and rejects boolean role."""
    # 3. credentials.kraken_credentials normalizes role case/whitespace and rejects boolean role
    pair = kraken_credentials(
        " BOT ",
        values={"KRAKEN_API_KEY_BOT": "k1", "KRAKEN_API_SECRET_BOT": "s1"},
    )
    assert pair.key == "k1"
    with pytest.raises(ValueError, match="Unknown Kraken credential role"):
        kraken_credentials(True, values={})

def test_botcore_http_request_fail_closed_on_malformed_method_url_iter99_s2():
    """[ITER99 #2] botcore.http_request fail-closed on malformed method/url."""
    # 2. botcore.http_request fail-closed on malformed method/url
    assert botcore.http_request(None, "https://example.com") == (0, b"")
    assert botcore.http_request(True, "https://example.com") == (0, b"")
    assert botcore.http_request("GET", None) == (0, b"")
    assert botcore.http_request("GET", True) == (0, b"")
    assert botcore.http_request("GET", "not-a-valid-url") == (0, b"")

def test_utils_timestamp_decay_helpers_iter99_s3():
    """[ITER99 #3] utils timestamp & decay helpers."""
    # 3. utils timestamp & decay helpers
    assert utils.timestampToTime(None) == "N/A"
    assert utils.timestampToTime(True) == "N/A"
    assert utils.timestampToTime(-100) == "N/A"
    assert utils.timestampToTime(float("nan")) == "N/A"
    assert utils.timeToHMS(None) == "N/A"
    assert utils.timeToHMS(False) == "N/A"
    assert utils.timeToHMS(-5) == "N/A"
    assert utils.timeToHMS(65).endswith(":01:05")

def test_providers_http_resilience_mount_connect_retry_rejects_boolean_status_forcelist_iter109_s2():
    """[ITER109 #2] providers.http_resilience: mount_connect_retry rejects boolean status_forcelist."""
    from providers import base as hres
    # 2. providers.http_resilience: mount_connect_retry rejects boolean status_forcelist
    assert hres.mount_connect_retry(SimpleNamespace(), status_forcelist=True) is False

def test_utils_value_diff_to_percent_secondsto_cache_path_iter110_s3():
    """[ITER110 #3] utils: value_diff_to_percent, secondsTo*, cache_path."""
    import utils as u
    # 3. utils: value_diff_to_percent, secondsTo*, cache_path
    assert u.value_diff_to_percent(True, 100.0) == 0.0
    assert u.value_diff_to_percent(float("nan"), 100.0) == 0.0
    assert u.value_diff_to_percent(0.0, 100.0) == 0.0
    assert u.secondsToDays(True) == 0.0
    assert u.secondsToHours(None) == 0.0
    assert u.cache_path(True) is True

def test_botcore__dotenv_pairs_required_env_defined_env_required_bool_env_iter114_s2():
    """[ITER114 #2] botcore._dotenv_pairs, required_env, defined_env, required_bool_env."""
    # 2. botcore._dotenv_pairs, required_env, defined_env, required_bool_env
    assert botcore._dotenv_pairs(None) == ([], False)
    assert botcore._dotenv_pairs(True) == ([], False)
    with pytest.raises(ValueError):
        botcore.required_env("API_KEY", {"API_KEY": True})
    with pytest.raises(ValueError):
        botcore.defined_env("API_KEY", {"API_KEY": False})
    assert botcore.required_bool_env("FLAG", {"FLAG": True}) is True
    assert botcore.required_bool_env("FLAG", {"FLAG": False}) is False

def test_jsonstatestore_normalizes_string_fail_closed_false_iter127_s4(tmp_path):
    """[ITER127 #4] JsonStateStore normalizes string fail_closed='false'."""
    # 4. JsonStateStore normalizes string fail_closed="false"
    store = JsonStateStore(
        str(tmp_path / "corrupt.json"),
        lambda: {"cycle": 1, "qty": 0.0},
        label="test",
        logger=lambda _m: None,
        fail_closed="false",
    )
    assert store.fail_closed is False

def test_jsonstatestore_fail_closed_boolean_parsing_iter155_s1(tmp_path):
    """[ITER155 #1] JsonStateStore fail_closed boolean parsing."""
    # 1. JsonStateStore fail_closed boolean parsing
    s_nan = JsonStateStore(str(tmp_path / "s.json"), dict, label="T", logger=lambda _: None, fail_closed=float("nan"))  # type: ignore[arg-type]
    assert s_nan.fail_closed is True
    s_str = JsonStateStore(str(tmp_path / "s.json"), dict, label="T", logger=lambda _: None, fail_closed="nan")  # type: ignore[arg-type]
    assert s_str.fail_closed is False

def test_botcore_rejects_none_pair_id_and_cleans_order_id_metadata_iter160_s2():
    """[ITER160 #2] botcore: rejects 'None'/'0' pair_id and cleans '.0'/'0' order_id metadata."""
    # 2. botcore: rejects "None"/"0" pair_id and cleans ".0"/"0" order_id metadata
    assert botcore.clean_order_id("None") is None
    assert botcore.clean_order_id("0") is None
    assert botcore.clean_order_id("0.0") is None
    assert botcore.clean_order_id(float("nan")) is None
    assert botcore.normalize_scalar_order_id("999.0") == "999"
    assert botcore.normalize_scalar_order_id("0") is None
    assert botcore.normalize_scalar_order_id("None") is None

def test_botcore_load_dotenv_override_false_does_not_overwrite_existing_iter162_s1(tmp_path, monkeypatch):
    """[ITER162 #1] botcore: load_dotenv(override='false') does not overwrite existing os.environ."""
    # 1. botcore: load_dotenv(override="false") does not overwrite existing os.environ
    env_file = tmp_path / "test.env"
    env_file.write_text("ITER162_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("ITER162_TEST_KEY", "from_process")
    botcore.load_dotenv(str(env_file), override="false")
    assert botcore.os.environ["ITER162_TEST_KEY"] == "from_process"
    botcore.load_dotenv(str(env_file), override="true")
    assert botcore.os.environ["ITER162_TEST_KEY"] == "from_file"

def test_config__parse_bool_flag_with_default_false_and__read_properties_file_with_iter162_s2(tmp_path):
    """[ITER162 #2] config: _parse_bool_flag with default='false' and _read_properties_file with quoted # and invalid path."""
    import botcore as cfg
    # 2. config: _parse_bool_flag with default="false" and _read_properties_file with quoted # and invalid path
    assert cfg._parse_bool_flag("unknown_token", default="false") is False
    assert cfg._read_properties_file(True) == {}
    prop_file = tmp_path / "props.env"
    prop_file.write_text('QUOTED_HASH="abc#def" # inline comment\nTRADE_ENABLED="false"\n', encoding="utf-8")
    props = cfg._read_properties_file(str(prop_file))
    assert props["QUOTED_HASH"] == "abc#def"
    assert props["TRADE_ENABLED"] is False

def test_credentials_resolve_credential_pair_validation_for_non_mapping_values_and_iter165_s1():
    """[ITER165 #1] credentials.resolve_credential_pair validation for non-Mapping values and malformed profiles."""
    import credentials
    # 1. credentials.resolve_credential_pair validation for non-Mapping values and malformed profiles
    with pytest.raises(credentials.CredentialConfigurationError):
        credentials.resolve_credential_pair([("Bot", "K", "S")], values=["not_a_mapping"])
    with pytest.raises(credentials.CredentialConfigurationError):
        credentials.resolve_credential_pair("not_a_sequence_of_tuples", values={})
    with pytest.raises(credentials.CredentialConfigurationError):
        credentials.resolve_credential_pair([], values={})
    with pytest.raises(credentials.CredentialConfigurationError):
        credentials.resolve_credential_pair([("Bot", "", "S")], values={})
    pair = credentials.resolve_credential_pair(
        [(" Bot ", " K ", " S ")],
        values={"K": " key_val ", "S": " sec_val "},
    )
    assert pair.key == "key_val" and pair.secret == "sec_val" and pair.profile == "Bot"

def test_providers_http_resilience_mount_connect_retry_with_string_numbers_and_raise_on_status_iter168_s3():
    """[ITER168 #3] providers.http_resilience.mount_connect_retry with string numbers and raise_on_status='false'."""
    from providers import base as hr
    # 3. providers.http_resilience.mount_connect_retry with string numbers and raise_on_status="false"
    mounted = {}

    class _DummySession:
        def mount(self, prefix, adapter):
            mounted[prefix] = adapter

    sess = _DummySession()
    assert hr.mount_connect_retry(
        sess,
        attempts="3",
        backoff="0.5",
        read="2",
        status_forcelist=["502", 503],
        raise_on_status="false",
    ) is True
    assert "https://" in mounted
    assert mounted["https://"].max_retries.raise_on_status is False
    assert hr.mount_connect_retry(sess, status_forcelist="500") is False

def test_config_load_config_with_empty_trade_enabled_env_var_iter168_s4(tmp_path, monkeypatch):
    """[ITER168 #4] config.load_config with empty TRADE_ENABLED env var."""
    import botcore as cfg_mod
    # 4. config.load_config with empty TRADE_ENABLED env var
    monkeypatch.setenv("TRADE_ENABLED", "   ")
    monkeypatch.setattr(cfg_mod, "CONFIG_ENV_PATH", str(tmp_path / "nonexistent.env"))
    monkeypatch.setattr(cfg_mod, "CONFIG_TXT_PATH", str(tmp_path / "nonexistent.txt"))
    cfg_mod.load_config()
    assert cfg_mod.is_trade_enabled() is True

def test_is_interval_due_handles_elapsed_non_elapsed_future_clock_skew_iter182_s1():
    """[ITER182 #1] is_interval_due handles elapsed, non-elapsed, future clock skew, None/bool/NaN, and force."""
    from botcore import is_interval_due
    # 1. is_interval_due handles elapsed, non-elapsed, future clock skew, None/bool/NaN, and force
    assert is_interval_due(1050.0, 1000.0, 60.0) is False
    assert is_interval_due(1060.0, 1000.0, 60.0) is True
    assert is_interval_due(1050.0, 1000.0, 60.0, force=True) is True
    assert is_interval_due(990.0, 1000.0, 60.0) is True
    assert is_interval_due(1050.0, None, 60.0) is True
    assert is_interval_due(1050.0, float("nan"), 60.0) is True

def test_resolve_order_notional_handles_explicit_notional_price_qty_fallback_and_iter182_s2():
    """[ITER182 #2] resolve_order_notional handles explicit notional, price*qty fallback, and nonfinite behavior."""
    from botcore import resolve_order_notional
    # 2. resolve_order_notional handles explicit notional, price*qty fallback, and nonfinite behavior
    assert resolve_order_notional(800.0, 100.0, 2.0) == pytest.approx(800.0)
    assert resolve_order_notional(None, 100.0, 2.5) == pytest.approx(250.0)
    assert resolve_order_notional(0.0, 100.0, 2.5) == pytest.approx(250.0)
    assert resolve_order_notional(None, 0.0, 2.5, default=None) is None
    assert resolve_order_notional(float("nan"), 100.0, 10.0, default=0.0, fallback_on_nonfinite=False) == pytest.approx(0.0)
