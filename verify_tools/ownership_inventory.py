#!/usr/bin/env python3
"""Read-only inventory of processes that can execute orders.

It neither reads nor displays API keys and does not block orders. It correlates
the process manifest with existing gates/configuration and flags only two active
``primary`` owners on the same venue/account/symbol. Protective processes
(trailing) are shown as intentional overlaps rather than conflicts.
"""

from __future__ import annotations

import argparse
import ast
import configparser
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from botcore import parse_dotenv  # noqa: E402
from providers.quantity import resolve_assets  # noqa: E402


_TRUE = {"1", "yes", "true", "on", "da"}


@dataclass(frozen=True)
class Owner:
    owner_id: str
    venue: str
    account_ref: str
    symbol: str
    base: str
    quote: str
    role: str
    coordination: str
    process_pattern: str | None
    configured: bool
    live_enabled: bool
    running: bool | None
    source: str

    def active(self, *, require_running: bool) -> bool:
        if require_running:
            return self.live_enabled and self.running is True
        return self.configured and self.live_enabled


def _truthy(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE


def _env(*paths: Path) -> dict[str, str]:
    """Later paths have higher priority; values are never reported."""
    result: dict[str, str] = {}
    for path in paths:
        result.update(parse_dotenv(str(path)))
    return result


def _account_ref(
    env: dict[str, str], venue: str, owner_id: str | None = None,
) -> str:
    owner_key = (
        re.sub(r"[^A-Z0-9]+", "_", owner_id.upper()).strip("_") + "_ACCOUNT_REF"
        if owner_id else ""
    )
    return (
        (env.get(owner_key) if owner_key else None)
        or env.get(f"{venue.upper()}_ACCOUNT_REF")
        or env.get("OWNERSHIP_ACCOUNT_REF")
        or f"{venue}:default"
    ).strip()


def _manifest(root: Path) -> list[str]:
    path = root / "procs.conf"
    if not path.exists():
        return []
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) >= 7 and parts[6].strip() in {"bot", "fleet"}:
            patterns.append(parts[0].strip())
    return patterns


def _pattern(patterns: list[str], target: str) -> str | None:
    return next((pattern for pattern in patterns if pattern == target), None)


def _commands() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="], check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _running(pattern: str | None, commands: list[str] | None) -> bool | None:
    if commands is None:
        return None
    if pattern is None:
        return False
    try:
        return any(re.search(pattern, command) for command in commands)
    except re.error:
        marker = pattern.rstrip("$")
        return any(marker in command for command in commands)


def _split_symbol(symbol: str, base: str = "", quote: str = "") -> tuple[str, str]:
    resolved_base, resolved_quote = resolve_assets(
        symbol, base=base or None, quote=quote or None)
    return resolved_base, resolved_quote or ""


def _python_symbols(root: Path) -> tuple[list[str], str]:
    """Read constants without importing `symbols.py` or triggering its API side effects."""
    path = root / "symbols.py"
    if not path.exists():
        return [], ""
    values = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return list(values.get("symbols") or []), str(values.get("taosymbol") or "")


def _record(
    *, owner_id: str, venue: str, account_ref: str, symbol: str,
    role: str, coordination: str, pattern: str | None, enabled: bool,
    source: str, commands: list[str] | None, base: str = "", quote: str = "",
    configured: bool | None = None,
) -> Owner:
    base, quote = _split_symbol(symbol, base, quote)
    return Owner(
        owner_id=owner_id,
        venue=venue,
        account_ref=account_ref,
        symbol=symbol,
        base=base,
        quote=quote,
        role=role,
        coordination=coordination,
        process_pattern=pattern,
        configured=pattern is not None if configured is None else configured,
        live_enabled=enabled,
        running=_running(pattern, commands),
        source=source,
    )


def build_inventory(root: Path = ROOT, commands: list[str] | None = None) -> list[Owner]:
    root = root.resolve()
    patterns = _manifest(root)
    owners: list[Owner] = []

    root_env = _env(root / "config.env", root / ".env")
    trade_settings = _env(root / "config.txt")
    kraken_env = _env(root / "kraken" / "config.env", root / "kraken" / ".env")
    hl_env = _env(root / "hyperliquid" / "config.env", root / "hyperliquid" / ".env")
    t212_env = _env(root / "212trading" / ".env")

    kraken_pair = kraken_env.get("KRAKEN_PAIR", "HYPEUSD").split("#", 1)[0].strip()
    kraken_ref = _account_ref(kraken_env, "kraken", "kraken-spot-dca")
    kraken_bot = "kraken_bot.py"
    owners.append(_record(
        owner_id="kraken-spot-dca", venue="kraken", account_ref=kraken_ref,
        symbol=kraken_pair, role="primary", coordination="spot-dca",
        pattern=kraken_bot,
        configured=_pattern(patterns, kraken_bot) is not None,
        enabled=_truthy(kraken_env.get("STRAT_EXECUTE")),
        source="kraken/config.env", commands=commands,
    ))

    trailing_env = _env(
        root / "kraken" / "config.env", root / "kraken" / "trailing.conf",
        root / "kraken" / ".env",
    )
    owners.append(_record(
        owner_id="kraken-trailing", venue="kraken",
        account_ref=_account_ref(kraken_env, "kraken", "kraken-trailing"),
        symbol=kraken_pair, role="protective", coordination="free-balance-only",
        pattern="kraken/trailing_stop.py",
        configured=_pattern(patterns, "kraken/trailing_stop.py") is not None,
        enabled=_truthy(trailing_env.get("KRAKEN_TRAILING_ENABLED")),
        source="kraken/trailing.conf", commands=commands,
    ))

    dn_pair = hl_env.get("HL_COIN", "HYPE").strip() or "HYPE"
    owners.append(_record(
        owner_id="hyperliquid-dn", venue="hyperliquid",
        account_ref=_account_ref(hl_env, "hyperliquid", "hyperliquid-dn"),
        symbol=dn_pair,
        base=dn_pair, quote="USDC", role="primary", coordination="delta-neutral",
        pattern="dn_bot.py$",
        configured=_pattern(patterns, "dn_bot.py$") is not None,
        enabled=_truthy(hl_env.get("STRAT_EXECUTE")),
        source="hyperliquid/config.env", commands=commands,
    ))

    hl_token = (
        hl_env.get("HL_SPOT_TOKEN") or hl_env.get("HL_COIN") or "HYPE"
    ).strip()
    owners.append(_record(
        owner_id="hyperliquid-spot-dca", venue="hyperliquid",
        account_ref=_account_ref(
            hl_env, "hyperliquid", "hyperliquid-spot-dca",
        ),
        symbol=f"{hl_token}USDC", base=hl_token, quote="USDC",
        role="primary", coordination="spot-dca", pattern="hl_dca_bot.py",
        configured=_pattern(patterns, "hl_dca_bot.py") is not None,
        enabled=(
            _truthy(hl_env.get("STRAT_EXECUTE"))
            and _truthy(hl_env.get("HL_LIVE_ORDERS"))
        ),
        source="hyperliquid/hl_dca_bot.py", commands=commands,
    ))

    monitor_pattern = "monitortrades.py"
    monitor_configured = _pattern(patterns, monitor_pattern) is not None
    trade_enabled = _truthy(
        root_env.get("TRADE_ENABLED")
        or root_env.get("trade_enabled")
        or trade_settings.get("trade_enabled", "true")
    )
    instruments_path = root / "instruments.conf"
    if instruments_path.exists():
        parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        parser.read(instruments_path, encoding="utf-8")
        venue_envs = {
            "binance": root_env,
            "kraken": kraken_env,
            "hyperliquid": hl_env,
            "t212": t212_env,
        }
        live_gates = {
            "binance": True,
            "kraken": _truthy(kraken_env.get("KRAKEN_LIVE_ORDERS")),
            "hyperliquid": _truthy(hl_env.get("HL_LIVE_ORDERS")),
            "t212": _truthy(t212_env.get("T212_LIVE_ORDERS")),
        }
        for section in parser.sections():
            values = parser[section]
            venue = values.get("provider", "unknown").strip().lower()
            symbol = values.get("symbol", section).strip()
            enabled = (
                _truthy(values.get("enabled"), True)
                and trade_enabled
                and live_gates.get(venue, False)
            )
            env = venue_envs.get(venue, {})
            owners.append(_record(
                owner_id=f"monitortrades:{section}", venue=venue,
                account_ref=(
                    values.get("ownership.account_ref", "").strip()
                    or _account_ref(env, venue, "monitortrades")
                ),
                symbol=symbol,
                base=values.get("base", "").strip(),
                quote=values.get("quote", "").strip(), role="primary",
                coordination=(
                    "binance-order-pipeline" if venue == "binance" else "market-api"
                ),
                pattern=monitor_pattern, configured=monitor_configured,
                enabled=enabled,
                source=f"instruments.conf[{section}]", commands=commands,
            ))

    t212_pattern = "t212_bot.py"
    t212_configured = _pattern(patterns, t212_pattern) is not None
    for path in sorted((root / "212trading").glob("config.*.env")):
        profile = path.name[len("config."):-len(".env")]
        profile_env = _env(path)
        symbol = profile_env.get("T212_TICKER", profile).split("#", 1)[0].strip()
        owners.append(_record(
            owner_id=f"t212:{profile}", venue="t212",
            account_ref=_account_ref({**profile_env, **t212_env}, "t212"),
            symbol=symbol, role="primary", coordination="t212-bot",
            pattern=t212_pattern, configured=t212_configured,
            enabled=(
                _truthy(profile_env.get("STRAT_ENABLED"))
                and _truthy(profile_env.get("STRAT_EXECUTE"))
            ),
            source=f"212trading/{path.name}", commands=commands,
        ))

    symbols, tao_symbol = _python_symbols(root)
    binance_ref = _account_ref(root_env, "binance")
    binance_specs = (
        ("tradeall", "tradeall.py", symbols, "primary"),
        ("rtrade", "rtrade.py", [tao_symbol] if tao_symbol else [], "primary"),
        ("assetguardian", "assetguardian.py", symbols, "protective"),
        ("binance-trailing", "binance_api/trailing_stop.py", symbols, "protective"),
    )
    for owner_id, process, owned_symbols, role in binance_specs:
        configured = _pattern(patterns, process) is not None
        enabled = True
        if owner_id == "binance-trailing":
            trailing = _env(root / "binance_api" / "trailing.conf", root / ".env")
            enabled = _truthy(trailing.get("TRAILING_ENABLED"))
        for symbol in owned_symbols:
            owners.append(_record(
                owner_id=owner_id, venue="binance", account_ref=binance_ref,
                symbol=symbol, role=role, coordination="binance-order-pipeline",
                pattern=process, configured=configured, enabled=enabled,
                source=process, commands=commands,
            ))

    return sorted(owners, key=lambda item: (
        item.venue, item.account_ref, item.symbol, item.owner_id,
    ))


def find_overlaps(owners: list[Owner], *, require_running: bool = False) -> list[dict]:
    groups: dict[tuple[str, str, str], list[Owner]] = {}
    for owner in owners:
        if owner.active(require_running=require_running):
            groups.setdefault(
                (owner.venue, owner.account_ref, owner.symbol), []
            ).append(owner)

    overlaps = []
    for (venue, account_ref, symbol), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        primary = [owner for owner in group if owner.role == "primary"]
        primary_domains = sorted({owner.coordination for owner in primary})
        overlaps.append({
            "venue": venue,
            "account_ref": account_ref,
            "symbol": symbol,
            "severity": (
                "warning"
                if len(primary) > 1 and len(primary_domains) > 1
                else "info"
            ),
            "owners": [owner.owner_id for owner in group],
            "primary_owners": [owner.owner_id for owner in primary],
            "primary_coordination_domains": primary_domains,
        })
    return overlaps


def _print(owners: list[Owner], *, require_running: bool) -> None:
    print(
        "Scope: processes running right now"
        if require_running else
        "Scope: versioned/local configuration (the process may not be running right now)"
    )
    print("venue         account_ref          symbol          owner                         role        active")
    print("------------- -------------------- --------------- ----------------------------- ----------- ------")
    for owner in owners:
        active = "yes" if owner.active(require_running=require_running) else "no"
        print(
            f"{owner.venue:<13} {owner.account_ref:<20} {owner.symbol:<15} "
            f"{owner.owner_id:<29} {owner.role:<11} {active}"
        )
    overlaps = find_overlaps(owners, require_running=require_running)
    print("\nOverlaps:")
    if not overlaps:
        print("  none")
    for item in overlaps:
        owners_text = ", ".join(item["owners"])
        print(
            f"  {item['severity'].upper():<7} {item['venue']}/"
            f"{item['account_ref']}/{item['symbol']}: {owners_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--running", action="store_true",
        help="treat an owner as active only if its process is running right now",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    commands = _commands() if args.running else None
    owners = build_inventory(args.root, commands=commands)
    overlaps = find_overlaps(owners, require_running=args.running)
    if args.as_json:
        print(json.dumps({
            "scope": "running" if args.running else "configured",
            "owners": [
                {**asdict(owner), "active": owner.active(require_running=args.running)}
                for owner in owners
            ],
            "overlaps": overlaps,
        }, indent=2, ensure_ascii=False))
    else:
        _print(owners, require_running=args.running)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
