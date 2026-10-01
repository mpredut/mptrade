from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_trailing_once_commands_share_daemon_lock():
    for relative, lock_name in (
        ("binance_api/trailing_stop.py", "binance_trailing"),
        ("kraken/trailing_stop.py", "kraken_trailing"),
    ):
        text = _text(relative)
        assert "if not args.status:" in text
        assert f'single_instance("{lock_name}")' in text


def test_xstock_once_shares_lock_but_isolated_trial_does_not():
    text = _text("kraken/kraken_xstock_watch.py")
    assert "if not args.status and not args.trial:" in text
    assert 'single_instance("kraken_xstock_watch")' in text


def test_strategy_test_commands_use_instrument_daemon_locks():
    kraken = _text("kraken/kraken_bot.py")
    hyperliquid = _text("hyperliquid/hl_bot.py")
    assert 'single_instance(f"kraken_bot_{args.test_strategy.strip()}")' in kraken
    assert 'single_instance(f"hl_bot_{args.test_strategy.strip()}")' in hyperliquid
    assert 'single_instance(f"hl_bot_{token}")' in hyperliquid
