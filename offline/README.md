# Offline tooling

This directory contains only tools that are not part of the live runtime.

Structure:

- `backtests/` -- replay/backtest engines that reuse the runtime code.
- `research/` -- experiments and research documentation.
- `runners/` -- orchestration of the backtests and of the prod -> dev flow.
- `manual/` -- diagnostics launched explicitly by an operator; some reach real APIs.
- `simulations/` -- local experiments, with no role in starting the fleet.
- `legacy_tools/` -- historical utilities and manual migrations, kept for auditing.

No runtime module may import code from here. Offline code may import runtime engines and
strategies, but the reverse dependency is forbidden. Scripts that can reach real accounts
must be run explicitly and checked before use.
