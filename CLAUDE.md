# Project conventions

## Language: English only

**All code, comments, docstrings, log messages, documentation and commit messages
must be written in English.** This applies to new files and to any line you touch
in an existing one — no exceptions, in any Claude session.

The migration is finished, everywhere. The live code, the shell scripts, the tests
(unittest method names included), the configuration comments, the research documents,
the archived and dead code, the ntfy alert bodies and the argparse help strings are all
English.

Exactly two things stay Romanian, and both for a mechanical reason:

- the word list in `verify_tools/translation_audit.py` — it is the scanner's own detection
  vocabulary, so translating it would blind the tool to what it exists to find;
- the identifier `prag` in `strategies/spot_dca_rules.py`, `strategies/spot_dca.py` and
  `212trading/strategy.py` — a variable name, not prose. Renaming it is a refactor of live
  trading code, not a translation.

Two consequences worth remembering, both learned the hard way:

- **after translating any string, run the test suite.** Tests match live wording through
  `assertRaisesRegex`/`assertIn`, so a translated message can break a test that has
  nothing to do with the file you edited.
- **patterns that match log or alert text are a separate case, and stay BILINGUAL.**
  `verify_tools/watchdogfor_anomaly.py` (the auth and blind regexes) and
  `alertnotifiers.py` (`_URGENT_MARKERS`, `_GUARD_MARKERS`, `_OPS_MARKERS`) keep their
  Romanian alternatives DELIBERATELY — owner's decision. The code is English, but a log
  line or an alert title could still be written in Romanian by mistake, and an
  English-only pattern would then fail silently: the watchdog would stop firing, and the
  alert would be routed to the routine topic and skip the email, with no error anywhere.
  Do not "clean these up" — they are the one place where over-matching is the safe side.

Anything else in Romanian is a regression. **Do not imitate it** — translate it.

## No Romanian diacritics, anywhere

Never write Romanian diacritics into a file in this repository: write `a` for `ă`/`â`,
`i` for `î`, `s` for `s-comma`, `t` for `t-comma`.

Exactly two files legitimately contain them, and neither is a mistake to fix: the
diacritic set inside `verify_tools/translation_audit.py`, which is the detector itself,
and the line above, which has to name the characters it forbids.

The files travel between Windows, WSL and the Linux server through plink, PowerShell and
9P mounts, and diacritics are the first thing those layers mangle. A mangled byte in a
`.sh` or a `.conf` is a runtime failure, not a cosmetic one.

Useful commands:

```bash
verify_tools/translation_audit.py scan            # what still looks Romanian
verify_tools/translation_audit.py verify <file>   # prove an edit touched prose only
```

`scan` still flags module docstrings that are reused as runtime CLI text (argparse).
Those are now English like everything else, but the flag remains useful: translating
one of them changes what the operator sees, so it is a behaviour change, not a cleanup.

## Deployment

Edit locally, commit, push to `main`, then `git pull --ff-only` on the server. No
restart mechanism ever pulls code, so a single pull is enough: every supervisor
(systemd, healthcheck) relaunches from what is already on disk.

See `systemd/PIA.md` for the VPN, and `systemd/README.md` for rebuilding PROD.
