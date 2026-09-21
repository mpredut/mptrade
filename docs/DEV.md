# The DEV / backtest profile

The authoritative configuration is `offline/runners/dev_backtest.env`; the host, the port,
the checkout and the branches are read exclusively from there.

DEV needs no cron and no live services of its own. PROD orchestrates:

- `refresh_dev.sh` every 30 minutes: fast-forward `main` and rsync `cachedb/`;
- `trigger_backtest_dev.sh` at 02:00 and 14:00;
- DEV runs `run_backtest_cycle.sh` and publishes to `backtest-proposals`;
- PROD applies the proposals with the guardrails in `apply_proposals.py`.

Rebuilding DEV: clone `main` into the configured path, create `myenv`, install
`requirements.txt`, authorise PROD's SSH key, and check `run_backtest_cycle.sh` by hand. Do
not install `systemd/install_prod.sh` on DEV and do not copy the PROD exchange keys; only
the synchronised `cachedb/` data is needed.
