import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

# --- PORTABILITY TESTS ---
DEPLOY_FILES = [
    "env_common.sh",
    "tools/admin/manage_backups.sh",
    "restart_bots.sh",
    "deploy_providers.sh",
    "tools/lib/process_control.sh",
    "fleet_supervisor.sh",
    "healthcheck.sh",
    "tools/monitoring/local_watch_start.sh",
    "tools/admin/manage_logs.sh",
    "tools/admin/pia_selfheal.sh",
    "pia_supervisor.sh",
    "systemd/PIA.md",
    "systemd/README.md",
    "systemd/bashrc",
    "systemd/binance.service",
    "systemd/crontab.prod.txt",
    "systemd/crontab.root.prod.txt",
    "systemd/install_prod.sh",
    "systemd/pia.service",
    "systemd/sudo.txt",
    "systemd/sudoers-trading",
    "systemd/trading-admin",
    "hyperliquid/hl-dn.service",
    "kraken/xstock-watch.service",
]


def test_deploy_files_do_not_depend_on_developer_accounts_or_checkout_name():
    forbidden = ("mariusp", "predut", "/home/", "~/binance", "$HOME/binance")
    offenders = {}
    for relative in DEPLOY_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8").lower()
        matches = [value for value in forbidden if value.lower() in text]
        if matches:
            offenders[relative] = matches
    assert offenders == {}


def test_deploy_templates_render_for_the_current_checkout(tmp_path):
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        python = Path(os.environ.get("PYTHON", os.sys.executable))
    env = {
        **os.environ,
        "TRADING_ROOT": str(ROOT),
        "TRADING_USER": subprocess.check_output(
            ["id", "-un"], text=True,
        ).strip(),
        "TRADING_PYTHON": str(python),
        "TRADING_RENDER_DIR": str(tmp_path),
    }
    subprocess.run(
        ["bash", str(ROOT / "systemd/install_prod.sh"), "--render-only"],
        check=True,
        cwd=ROOT,
        env=env,
    )

    rendered = list(tmp_path.iterdir())
    assert rendered
    for path in rendered:
        text = path.read_text(encoding="utf-8")
        assert "@TRADING_" not in text
    assert f"WorkingDirectory={ROOT}" in (tmp_path / "binance.service").read_text()
    assert "TRADING_SUPERVISE_ENABLED=true" in (
        tmp_path / "crontab.prod.txt"
    ).read_text()
    rendered_crontab = (tmp_path / "crontab.prod.txt").read_text()
    portfolio_line = next(
        line for line in rendered_crontab.splitlines()
        if "portfolio_snapshot.py" in line
    )
    assert "/usr/bin/flock -n" in portfolio_line
    assert "/usr/bin/timeout --signal=TERM --kill-after=5s 90s" in portfolio_line

# --- REFRESH TESTS ---


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def sandbox(tmp_path):
    # deploy_providers.sh, restart_bots.sh, tools/lib/process_control.sh, env_common.sh
    for name in ("deploy_providers.sh", "restart_bots.sh", "env_common.sh"):
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / "tools" / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "tools/lib/process_control.sh", tmp_path / "tools/lib/process_control.sh")
    (tmp_path / "procs.conf").write_text(
        "cacheManager.py|$ROOT||cache|||fleet\n"
        "binance_api/trailing_stop.py|$ROOT|exit 99|trailing|||bot\n")
    executable(tmp_path / ".venv/bin/python", "#!/bin/sh\nexit 0\n")
    (tmp_path / ".venv/bin/activate").touch()
    for command in ("git", "kill", "pkill", "pgrep"):
        executable(tmp_path / "fake-bin" / command, "#!/bin/sh\necho unexpected-command >&2\nexit 91\n")
    return dict(os.environ, PATH=f"{tmp_path / 'fake-bin'}:{os.environ['PATH']}")


def test_check_is_offline_and_includes_independent_trailing(tmp_path):
    env = sandbox(tmp_path)
    # pgrep is allowed in read-only mode; it reports no local trading processes.
    executable(tmp_path / "fake-bin/pgrep", "#!/bin/sh\nexit 1\n")
    result = subprocess.run(["bash", "deploy_providers.sh", "--check"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "role=fleet" in result.stdout and "role=bot" in result.stdout
    assert "unexpected-command" not in result.stderr


def test_failed_pull_never_restarts(tmp_path):
    env = sandbox(tmp_path)
    result = subprocess.run(["bash", "deploy_providers.sh"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 91
    assert "role=" not in result.stdout


def test_successful_deploy_refreshes_both_roles_before_reporting_success(tmp_path):
    env = sandbox(tmp_path)
    executable(tmp_path / "fake-bin/git", "#!/bin/sh\nexit 0\n")
    executable(tmp_path / "fake-bin/sleep", "#!/bin/sh\nexit 0\n")
    executable(tmp_path / "fake-bin/ps", "#!/bin/sh\necho S\n")
    # Synthetic inventory and launchers: no real process is signalled or launched.
    (tmp_path / "tools/lib/process_control.sh").write_text('''
manifest_pids() { if [ -f "$ROOT/bots-refreshed" ]; then echo 7; fi; }
stop_manifest_process() { printf '%s\\n' "$1" >> "$ROOT/fleet-refreshed"; }
''')
    (tmp_path / "restart_bots.sh").write_text(
        'touch "$(dirname "$0")/bots-refreshed"\necho launcher-ready\n')
    result = subprocess.run(["bash", "deploy_providers.sh"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "fleet-refreshed").read_text().strip() == "cacheManager.py"
    assert (tmp_path / "bots-refreshed").exists()
    assert (tmp_path / "logs/deploy_restart_bots.log").read_text().strip() == "launcher-ready"
    assert "Deployment verified" in result.stdout


def test_process_matching_and_stop_are_scoped_to_checkout(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    # Both command lines match the manifest pattern, but only one checkout is in scope.
    args = [sys.executable, "-c", "import time; time.sleep(30)", "registry_test_bot.py"]
    a = subprocess.Popen(args, cwd=first)
    b = subprocess.Popen(args, cwd=second)
    try:
        found = subprocess.check_output(
            ["bash", "-c", 'source "$1"; manifest_pids registry_test_bot.py "$2"',
             "test", str(ROOT / "tools/lib/process_control.sh"), str(first)], text=True, timeout=10)
        assert found.split() == [str(a.pid)]
        # Reap the child concurrently so the graceful-stop test does not see a zombie.
        import threading
        waiter = threading.Thread(target=a.wait)
        waiter.start()
        stopped = subprocess.run(
            ["bash", "-c", 'source "$1"; stop_manifest_process registry_test_bot.py "$2"',
             "test", str(ROOT / "tools/lib/process_control.sh"), str(first)], timeout=15)
        waiter.join(timeout=2)
        assert stopped.returncode == 0
        assert a.poll() is not None and b.poll() is None
    finally:
        for process in (a, b):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

# --- HARDENING TESTS ---


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


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


@pytest.mark.parametrize(
    ("script", "var_name"),
    [
        ("pia_supervisor.sh", "CLI_TIMEOUT"),
        ("fleet_supervisor.sh", "PIA_CLI_TIMEOUT"),
        ("healthcheck.sh", "PIA_CLI_TIMEOUT"),
    ],
)
def test_hardcoded_piactl_timeouts(script, var_name):
    text = _text(script)
    assert f"{var_name}=" in text
    assert 'timeout "$' in text


def test_alert_transports_reject_http_errors():
    assert "--fail-with-body" in _text("tools/monitoring/deadman_switch.sh")
    assert "--fail-with-body" in _text("tools/admin/pia_selfheal.sh")
    assert "--fail-with-body" in _text("healthcheck.sh")


def test_installer_is_verified_with_pinned_sha256():
    text = _text("tools/admin/pia_selfheal.sh")
    assert "INSTALLER_SHA256=" in text
    assert 'sha256sum "$tmp/pia.run"' in text
    assert 'actual_sha256" != "$INSTALLER_SHA256' in text


def test_spool_is_drained_one_confirmed_alert_at_a_time():
    text = _text("tools/admin/pia_selfheal.sh")
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
    text = _text("tools/admin/pia_selfheal.sh")
    config = _text("config.env")
    assert 'CONFIG="$ROOT/config.env"' in text
    assert "resolved_cpu_percent" in text
    assert "PIA_RESOLVED_CPU_CONSECUTIVE" in config
    assert "systemctl restart systemd-resolved.service" in text


def test_selfheal_restarts_a_downed_fleet():
    # A clean stop of binance.service does not trigger its own Restart=always, so the
    # PIA-healthy path must bring the fleet back (guarded by a maintenance pause flag).
    text = _text("tools/admin/pia_selfheal.sh")
    assert "systemctl start binance.service" in text
    assert "FLEET_PAUSED" in text


def test_autodeploy_is_shadow_by_default_and_never_reboots():
    text = _text("tools/admin/git_autodeploy.sh")
    assert "AUTODEPLOY_MODE=shadow" in text          # default is observe-only
    assert "status --porcelain" in text              # dirty-tree guard
    assert 'merge-base HEAD "origin/$BRANCH"' in text # fast-forward-only guard
    assert "systemctl restart binance.service" in text
    assert "systemctl reboot" not in text            # never reboots the machine


def test_deadman_has_an_independent_healthchecks_channel():
    # ntfy's shared free topic got 429-throttled during the incident; the deadman must also
    # ping a dedicated healthchecks.io URL (if configured) so the alarm survives that.
    text = _text("tools/monitoring/deadman_switch.sh")
    assert "HC_PING_URL" in text
