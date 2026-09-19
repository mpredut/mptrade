"""Offline deployment checks and scoped process-control tests; never start trading bots."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


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
