"""config.env is sourced by root shell daemons (pia_supervisor.sh, vpn_watchdog.sh) under `set -u` with no HOME.

Any `$VAR` reference in it silently kills those daemons (unbound variable with stderr discarded), so the file
must be sourceable from an empty environment. Use `~/...` for user paths (expanded by expand_config_path).
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_env_sources_under_set_u_with_empty_environment():
    proc = subprocess.run(
        ["/bin/bash", "-c", "set -u; set -a; . ./config.env && echo SOURCE_OK"],
        cwd=ROOT, env={}, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0 and "SOURCE_OK" in proc.stdout, proc.stderr


def test_config_module_parses_falsy_trade_enabled_fail_closed(tmp_path, monkeypatch):
    import config

    env_file = tmp_path / "config.env"
    txt_file = tmp_path / "config.txt"
    monkeypatch.setattr(config, "CONFIG_ENV_PATH", str(env_file))
    monkeypatch.setattr(config, "CONFIG_TXT_PATH", str(txt_file))
    monkeypatch.delenv("TRADE_ENABLED", raising=False)

    for raw in ("0", "false", '"false"', "'0'", "no", "off"):
        env_file.write_text(f"export TRADE_ENABLED={raw}\n", encoding="utf-8")
        config.load_config()
        assert config.is_trade_enabled() is False, f"Expected False for {raw}"

    env_file.write_text("TRADE_ENABLED=1\n", encoding="utf-8")
    txt_file.write_text("trade_enabled=0 # override\n", encoding="utf-8")
    config.load_config()
    assert config.is_trade_enabled() is False

    config.config_cache["trade_enabled"] = "0"
    assert config.is_trade_enabled() is False

    config.start_config_watcher()
    assert config.watcher_thread is not None and config.watcher_thread.is_alive()
    config.stop_config_watcher()
    assert config.watcher_thread is None


def test_shell_launchers_source_env_common_correctly():
    with open(os.path.join(ROOT, "kraken", "spcx_launch.sh"), encoding="utf-8") as fh:
        spcx_text = fh.read()
    assert 'source "$HERE/../orchestratorOS/lib/env_common.sh"\n\nSTRAT_ENTRY=800' in spcx_text

    with open(
        os.path.join(ROOT, "hyperliquid", "archive", "delta_neutral", "dn_close.sh"),
        encoding="utf-8",
    ) as fh:
        dn_text = fh.read()
    assert 'source "$HERE/../../../orchestratorOS/lib/env_common.sh"' in dn_text
