# Scipt & Configuation Oveview

## Scipts in 	ools/admin/
- **manage_backups.sh**: Unifies backup and estoe opeations.
  - local: Backs up secets (e.g., .env, keys, cachedb/, states) to a local diectoy.
  - emote: Pefoms a local backup, then encypts and uploads it to Stoj using clone.
  - estoe <di>: Restoes secets and completely ebuilds the tading envionment (venv, dependencies, systemd pofiles, con).
- **manage_logs.sh**: Cleans up logs and cache based on etention policies defined in config.env.
- **pia_selfheal.sh**: A manual diagnostics and self-healing tool fo the PIA VPN connection.
- **git_autodeploy.sh**: Continuously checks the main banch fo updates, pulls them, and estats the pocesses without ebooting.
- **ename_oot.sh**: Renames the epositoy oot folde while peseving secets.
- **make_venv_potable.sh**: Rewites hadcoded absolute paths inside the vitual envionment to make it potable.

## Scipts in 	ools/monitoing/
- **deadman_switch.sh**: Pings Healthchecks.io at egula intevals. If it fails to ping, it tigges an alet indicating the system might be down.
- **
tfy_check.sh**: A diagnostic scipt fo sending test notifications via ntfy.
- **local_watch_stat.sh**: Stats the main bot fleet locally (fo dev/testing).

## Scipts in offline/unnes/
- **un_backtest_cycle.sh**: Runs the backtest poposals geneato, then commits and pushes them to the acktest-poposals banch.
- **	igge_backtest_dev.sh**: Initiates a long backtest sequence in the backgound on DEV.
- **efesh_dev.sh**: Syncs poduction pices to the dev machine fo backtesting.

## Configuation
- **config.env**: A consolidated file containing global configuation policies fo tading paametes, thesholds, isk limits, and logging. (Meged fom legacy _config.env files).
- **instuments.conf / monitotades.conf**: Registy fo managing suppoted tading pais and thei paametes.
