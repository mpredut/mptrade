#!/usr/bin/env python3
"""
trailing_stop.py (Kraken) -- crash circuit breaker for Kraken spot holdings.

Thin Kraken venue entrypoint backed by the shared
``trailing_core.StrategyExecutorTrailingAdapter`` and ``TrailingCore`` state machine.
Configuration lives in ``instruments.conf`` (provider=kraken, role.trailing=yes)
and ``kraken/trailing.conf``. See ``tests/test_trailing_kraken.py``.

  python3 trailing_stop.py        # loop (enabled in trailing.conf)
  python3 trailing_stop.py --once # one check
  python3 trailing_stop.py --status
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from botcore import (  # noqa: E402
    extract_order_side,
    load_dotenv,
    load_env_stack,
    log,
    required_bool_env,
    required_float_env,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (  # noqa: E402
    CACHE_SHORT_TREND_FILENAME,
    cache_path,
    kraken_pair_matches,
    resolve_venue_trailing_state_path,
    split_kraken_pair,
)
from credentials import kraken_credentials  # noqa: E402
from kraken_client import KrakenClient  # noqa: E402
from kraken_common import build_kraken_client, notify  # noqa: E402
from providers.kraken_provider import (  # noqa: E402
    KrakenProvider,
    _kraken_balance_asset_candidates,
)
from trailing_core import (  # noqa: E402
    StrategyExecutorTrailingAdapter,
    load_trailing_common_env,
    load_venue_trailing_registry,
    parse_trailing_cli_args,
)

STATE_FILE = resolve_venue_trailing_state_path("kraken")
CACHE_TREND = cache_path(CACHE_SHORT_TREND_FILENAME)

# Match launch order used by every Kraken process: real env, secrets, versioned policy,
# then dedicated trailing policy layer.
load_env_stack(os.path.join(_HERE, ".env"))
load_dotenv(os.path.join(_HERE, "trailing.conf"))

DEFAULT_TRAIL_PCT = required_float_env("KRAKEN_TRAILING_DEFAULT_PCT")
MIN_NOTIONAL_USD = required_float_env("KRAKEN_TRAILING_MIN_NOTIONAL_USD")
(
    CHECK_SECONDS,
    REBUY_ENABLED,
    REBUY_BOUNCE_PCT,
    REBUY_SKIP_IF_TREND_DOWN,
    SELL_SKIP_IF_TREND_UP,
    SELL_TREND_DEFER_MAX_SEC,
    MIN_PROFIT_PCT,
) = load_trailing_common_env("KRAKEN_TRAILING_")
TRAILING_MARKET = required_bool_env("KRAKEN_TRAILING_MARKET")

(
    TRAILING_INSTRUMENTS,
    TRAIL_PCT,
    PAIR_FOR,
    REBUY_MODE_BY_ASSET,
    QUOTE_FOR,
) = load_venue_trailing_registry(
    "kraken",
    env_prefix="KRAKEN_TRAILING_",
    key_by_asset=True,
    default_trail_pct=DEFAULT_TRAIL_PCT,
)


class KrakenTrailing(StrategyExecutorTrailingAdapter):
    """Thin Kraken adapter delegating shared trailing lifecycle logic to ``StrategyExecutorTrailingAdapter``."""

    _VENUE_SLUG = "kraken"
    _VENUE_LABEL = "Kraken"
    _HEARTBEAT_NAME = "Kraken-trailing"
    _LOG_TAG = "[TRAIL-K]"
    _ENABLED_ENV_VAR = "KRAKEN_TRAILING_ENABLED"

    def __init__(
        self,
        client: KrakenClient,
        log=log,
        enabled=None,
        state_file=STATE_FILE,
        min_profit_pct=MIN_PROFIT_PCT,
        trailing_market: bool | None = None,
    ):
        self.client = client
        self._init_executor_adapter(
            KrakenProvider(client=client),
            log=log,
            enabled=enabled,
            state_file=state_file,
            min_profit_pct=min_profit_pct,
            trailing_market=trailing_market,
            notify_fn=notify,
        )

    def _balance_ex_map(self) -> dict | None:
        if self._tick_balances is not None:
            return self._tick_balances
        try:
            bx_map = self.client._private("BalanceEx")
        except Exception as e:  # noqa: BLE001
            self.log(f"  ! {self._LOG_TAG} BalanceEx failed ({e.__class__.__name__}: {e})")
            return None
        if not isinstance(bx_map, dict) or not bx_map:
            return None
        return bx_map

    def _fetch_balances_snapshot(self) -> dict | None:
        return self._balance_ex_map()

    def _balance_ex_entries(self, asset: str) -> list[object]:
        if isinstance(asset, bool) or not isinstance(asset, str) or not asset.strip():
            return []
        bx_map = self._balance_ex_map()
        if not isinstance(bx_map, dict):
            return []
        cand_folded = {c.casefold() for c in _kraken_balance_asset_candidates(asset) if isinstance(c, str) and c}
        if not cand_folded:
            return []
        return [
            val
            for k, val in bx_map.items()
            if isinstance(k, str) and k.strip().casefold() in cand_folded
        ]

    def has_valid_balance(self, asset: str) -> bool:
        if self._balance_ex_map() is None:
            return False
        for bx in self._balance_ex_entries(asset):
            if not isinstance(bx, dict):
                return False
            bal_f = safe_nonneg_float(bx.get("balance"), default=None)
            hold_f = safe_nonneg_float(bx.get("hold_trade", 0.0), default=None)
            if bal_f is None or hold_f is None:
                return False
        return True

    def free_qty(self, asset: str) -> float:
        """Return FREE balance (total minus order holds), excluding locked orders."""
        total_free = 0.0
        for bx in self._balance_ex_entries(asset):
            if not isinstance(bx, dict):
                return 0.0
            bal_f = safe_nonneg_float(bx.get("balance"), default=None)
            hold_f = safe_nonneg_float(bx.get("hold_trade", 0.0), default=None)
            if bal_f is None or hold_f is None:
                return 0.0
            total_free += safe_nonneg_float(bal_f - hold_f, default=0.0)
        return total_free

    def total_qty(self, asset: str) -> float | None:
        if self._balance_ex_map() is None:
            return None
        total_bal = 0.0
        for bx in self._balance_ex_entries(asset):
            if not isinstance(bx, dict):
                return None
            bal_f = safe_nonneg_float(bx.get("balance", 0), default=None)
            if bal_f is None:
                return None
            total_bal += bal_f
        return total_bal

    def _asset_for_pair(self, pair: str) -> str:
        resolved = super()._asset_for_pair(pair)
        return resolved or split_kraken_pair(pair)[0]

    def _open_sell_orders_for_unlock(self, pair: str) -> list[str]:
        open_ords = self.client.open_orders() or {}
        if not isinstance(open_ords, dict):
            return []
        oids: list[str] = []
        for txid, ord_info in open_ords.items():
            if not isinstance(ord_info, dict):
                continue
            raw_descr = ord_info.get("descr")
            descr = raw_descr if isinstance(raw_descr, dict) else {}
            ord_pair = str(descr.get("pair") or ord_info.get("pair") or "")
            if extract_order_side(ord_info) == "SELL" and (ord_pair == pair or kraken_pair_matches(pair, ord_pair)):
                oids.append(str(txid))
        return oids

    def _cancel_open_order_for_unlock(self, pair: str, oid: str) -> bool:
        _ = pair
        res = self.client.cancel_order(oid)
        if res is False or (
            isinstance(res, dict)
            and "count" in res
            and safe_pos_float(res.get("count"), default=0.0) <= 0.0
        ):
            return False
        return True


def main() -> int:
    args = parse_trailing_cli_args("A trailing-stop circuit breaker on Kraken (with a re-buy).", "kraken_trailing")
    # A dedicated trailing key (KRAKEN_API_KEY_TRAIL) keeps its nonce separate from
    # _BOT (kraken_bot + xstock_watch). Fall back to _BOT when _TRAIL is absent.
    client = build_kraken_client(KrakenClient, kraken_credentials, "trail")
    ts = KrakenTrailing(client)
    return ts.dispatch_cli(args)


if __name__ == "__main__":
    sys.exit(main())
