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
