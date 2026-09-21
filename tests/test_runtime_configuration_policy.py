from pathlib import Path
import subprocess
import re


ROOT = Path(__file__).resolve().parents[1]


def test_market_alerts_and_hyperliquid_provider_use_shared_env_stack():
    market = (ROOT / "market_monitor/price_notifier.py").read_text(encoding="utf-8")
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
    remaining = {
        "order_retry.py",                    # locked durable JSONL outbox
    }
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
            r"\b(?:tmp|tmp_file|temporary)\s*=.*(?:\.tmp|mkstemp)", text,
        )
        if owns_temporary and ("os.replace" in text or "mkstemp" in text):
            candidates.append(relative)
    assert set(candidates) == remaining


def test_dead_trade_cache_manager_is_archived_not_packaged():
    assert (ROOT / "archive/tradeCacheManager.py").is_file()
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


def test_live_launcher_env_defaults_are_location_based():
    for relative in (
        "kraken/kraken_bot.py",
        "hyperliquid/dn_bot.py",
        "hyperliquid/hl_bot.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert 'os.environ.get("ENV_FILE", ".env")' not in text
        assert 'os.path.dirname(__file__), ".env"' in text


def test_live_venue_credentials_use_complete_profile_resolver():
    files = (
        "kraken/kraken_bot.py",
        "kraken/kraken_cachemanager.py",
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
