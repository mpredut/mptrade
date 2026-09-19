from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_FILES = [
    "env_common.sh",
    "tools/admin/backup_remote.sh",
    "tools/admin/backup_local.sh",
    "restart_bots.sh",
    "deploy_providers.sh",
    "tools/lib/process_control.sh",
    "fleet_supervisor.sh",
    "healthcheck.sh",
    "tools/monitoring/local_watch_start.sh",
    "tools/admin/manage_logs.sh",
    "tools/admin/pia_selfheal.sh",
    "pia_supervisor.sh",
    "tools/admin/restore.sh",
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
