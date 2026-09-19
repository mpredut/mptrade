import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _text(name):
    path = ROOT / name
    if not path.is_file() and name == "pia_selfheal.sh":
        path = ROOT / "tools/admin/pia_selfheal.sh"
    return path.read_text(encoding="utf-8")


def _executable(path, body):
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _healthcheck_env(tmp_path, failing=None):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    commands = {
        "piactl": "echo Connected",
        "ip": "echo '2: tun0: <POINTOPOINT,UP,LOWER_UP>'",
        "resolvectl": "exit 0",
        "curl": "exit 0",
    }
    if failing == "piactl":
        commands["piactl"] = "echo Disconnected"
    elif failing in commands:
        commands[failing] = "exit 1"
    for name, body in commands.items():
        _executable(fake_bin / name, body)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["PIA_CLI_TIMEOUT"] = "1"
    env["PIA_PROBE_TIMEOUT"] = "1"
    return env, fake_bin


def _run_healthcheck(tmp_path, failing=None, mode="--check"):
    env, fake_bin = _healthcheck_env(tmp_path, failing)
    result = subprocess.run(
        ["bash", str(ROOT / "healthcheck.sh"), mode],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return result, fake_bin


def test_all_runtime_piactl_calls_are_bounded():
    for name, timeout_var in (
        ("pia_start.sh", "CLI_TIMEOUT"),
        ("flota_start.sh", "PIA_CLI_TIMEOUT"),
    ):
        text = _text(name)
        assert f'timeout "${timeout_var}" piactl "$@"' in text
        runtime = "\n".join(
            line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        assert runtime.count("piactl") == 1, name


def test_alert_transports_reject_http_errors():
    assert "--fail-with-body" in _text("deadman_switch.sh")
    assert "--fail-with-body" in _text("pia_selfheal.sh")
    assert "--fail-with-body" in _text("healthcheck.sh")


def test_installer_is_verified_with_pinned_sha256():
    text = _text("pia_selfheal.sh")
    assert "INSTALLER_SHA256=" in text
    assert 'sha256sum "$tmp/pia.run"' in text
    assert 'actual_sha256" != "$INSTALLER_SHA256' in text


def test_spool_is_drained_one_confirmed_alert_at_a_time():
    text = _text("pia_selfheal.sh")
    assert 'IFS= read -r line < "$file"' in text
    assert 'ntfy_push "PIA: alerta intarziata' in text
    assert 'tail -n +2 "$file"' in text


def test_healthcheck_probes_the_required_vpn_path():
    text = _text("healthcheck.sh")
    assert 'timeout "$PIA_CLI_TIMEOUT" piactl "$@"' in text
    # The tunnel interface is wgpia0 under WireGuard (not tun0); it must be parameterised so
    # the probe matches the live interface, or healthcheck reports a permanent false VPN
    # fault (spamming alerts and masking a real outage).
    assert 'VPN_IF="${PIA_VPN_IF:-wgpia0}"' in text
    assert 'ip link show dev "$VPN_IF"' in text
    assert 'resolvectl query -i "$VPN_IF" api.binance.com' in text
    assert '--interface "$VPN_IF"' in text
    assert "https://api.binance.com/api/v3/time" in text
    assert 'VPN($vpn)' in text


@pytest.mark.parametrize(
    ("failing", "reason"),
    [("piactl", "piactl"), ("ip", "wgpia0"), ("resolvectl", "dns"), ("curl", "https")],
)
def test_healthcheck_reports_each_simulated_vpn_failure(tmp_path, failing, reason):
    result, _ = _run_healthcheck(tmp_path, failing)
    assert result.returncode == 0, result.stderr
    assert f"VPN              FAULT ({reason})" in result.stdout


def test_healthcheck_accepts_a_fully_working_simulated_vpn(tmp_path):
    result, _ = _run_healthcheck(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "VPN              ok (piactl + wgpia0 + DNS + Binance HTTPS)" in result.stdout


def test_healthcheck_surfaces_ntfy_http_failure(tmp_path):
    env, fake_bin = _healthcheck_env(tmp_path)
    _executable(fake_bin / "pgrep", "exit 1")
    _executable(fake_bin / "curl", "exit 22")
    result = subprocess.run(
        ["bash", str(ROOT / "healthcheck.sh"), "--alert"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ALERT NOT DELIVERED" in result.stdout


def test_selfheal_is_part_of_reproducible_root_cron():
    cron = _text("systemd/crontab.root.prod.txt")
    installer = _text("systemd/install_prod.sh")
    assert "pia_selfheal.sh" in cron
    assert 'render "$SYSTEMD_DIR/crontab.root.prod.txt"' in installer
    assert 'crontab -u root "$TMP_DIR/crontab.root.prod.txt"' in installer


def test_selfheal_watches_resolver_cpu_and_requires_versioned_policy():
    text = _text("pia_selfheal.sh")
    config = _text("pia_selfheal_config.env")
    assert 'CONFIG="$ROOT/pia_selfheal_config.env"' in text
    assert "resolved_cpu_percent" in text
    assert "PIA_RESOLVED_CPU_CONSECUTIVE" in config
    assert "systemctl restart systemd-resolved.service" in text


def test_selfheal_restarts_a_downed_fleet():
    # A clean stop of binance.service does not trigger its own Restart=always, so the
    # PIA-healthy path must bring the fleet back (guarded by a maintenance pause flag).
    text = _text("pia_selfheal.sh")
    assert "systemctl start binance.service" in text
    assert "FLEET_PAUSED" in text


def test_autodeploy_is_shadow_by_default_and_never_reboots():
    text = _text("git_autodeploy.sh")
    assert "AUTODEPLOY_MODE=shadow" in text          # default is observe-only
    assert "status --porcelain" in text              # dirty-tree guard
    assert 'merge-base HEAD "origin/$BRANCH"' in text # fast-forward-only guard
    assert "systemctl restart binance.service" in text
    assert "systemctl reboot" not in text            # never reboots the machine


def test_deadman_has_an_independent_healthchecks_channel():
    # ntfy's shared free topic got 429-throttled during the incident; the deadman must also
    # ping a dedicated healthchecks.io URL (if configured) so the alarm survives that.
    text = _text("deadman_switch.sh")
    assert "HC_PING_URL" in text
