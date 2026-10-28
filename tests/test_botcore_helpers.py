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

import alertnotifiers
import botcore
from botcore import load_env_stack, parse_dotenv
from cachepaths import (
    CACHEDB_SUBDOMAINS,
    cache_path,
    ensure_cachedb_subdomains,
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
             patch.object(alertnotifiers.AlertNotifier, "send_email_batch") as email, \
             patch.object(alertnotifiers.subprocess, "run") as desktop:
            alertnotifiers.notify(
                "STOP-LOSS test", "paper event", "pytest", "TSTX",
                desktop=True, email=True,
            )
        ntfy.assert_not_called()
        email.assert_not_called()
        desktop.assert_not_called()


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
            "btc_1s.npz": "prices",
            "cache_short_trend.json": "trends",
            "cache_long_trend.json.lock": "trends",
            "kalman_state.json": "trends",
            "24trend": "trends",
            "orderbook_depth_BTCUSDC.json": "telemetry",
            "derivatives_telemetry_BTCUSDC.json": "telemetry",
            "whale_snapshot_ETHUSDC.json": "telemetry",
            "fear_greed_index.json": "macro",
            "market_breadth_metrics.json": "macro",
            "news_feed_snapshot.json": "macro",
            "macro_lexicon_collect.json": "macro",
            "sentiment_advisor_eval.json": "macro",
            "geopolitical_threat_eval.json": "macro",
            "cache_order.json": "orders",
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
            "overnight_monitor_status.json": "system",
            "dev_regression_status.json": "system",
            "llm_topic_threads.json": "system",
            "notification_queue.jsonl": "system",
            "procs.conf.reload": "system",
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
                sorted(moved["system/heartbeat"]),
                ["macro_analyzer.heartbeat", "rtrade_BTCUSDC.heartbeat"],
            )
            self.assertFalse(stale_dir.exists())
            self.assertTrue((root / "prices" / "cache_current_price.json").exists())
            self.assertTrue((root / "trends" / "cache_short_trend.json").exists())
            self.assertTrue((root / "macro" / "fear_greed_index.json").exists())
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

    # _OrderedIdSet trim
    s = monitororder._OrderedIdSet()
    for i in range(10):
        s.add(str(i))
    assert len(s) == 10
    s.trim(5)
    assert len(s) == 5
    assert list(s) == ["5", "6", "7", "8", "9"]


def test_kraken_cache_touch_ws_fetchtimes():
    import kraken.kraken_cachemanager as kcm
    cache = {"items": {"TAOUSDC": []}, "fetchtime": {}}
    kcm._touch_ws_fetchtimes(cache, 1700000000)
    assert cache["fetchtime"]["TAOUSDC"] == 1700000000
    kcm._touch_ws_fetchtimes(cache, "invalid")
    assert cache["fetchtime"]["TAOUSDC"] > 0


if __name__ == "__main__":
    unittest.main()
