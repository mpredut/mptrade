#!/usr/bin/env python3
"""Reusable Variant-A capital rebalance tool: Kraken -> Arbitrum One -> Hyperliquid Spot.

Security architecture (Variant A -- zero master wallet private key on server):
  1. Uses the isolated ``Kraken TRANSFER`` credential profile
     (``KRAKEN_API_KEY_TRANSFER`` / ``KRAKEN_API_SECRET_TRANSFER`` in ``kraken/.env``).
  2. Never accepts raw unwhitelisted withdrawal destinations. Verifies that the selected
     Kraken withdrawal key is ``verified=True``, uses native USDC on Arbitrum One
     (``Arbitrum One Network - Native``, rejecting bridged ``USDC.e``), and matches the
     target Hyperliquid wallet address (``HL_ACCOUNT_ADDRESS`` in ``hyperliquid/.env``).
  3. Dry-run by default: real orders/withdrawals run ONLY with explicit ``--execute``.
  4. Enforces net receipt >= ``KRAKEN_HL_MIN_NET_USDC`` (6.0 USDC, above Hyperliquid
     Bridge2's 5.0 USDC hard loss floor) and checks Arbitrum One ETH gas on the wallet.
  5. Optional ``--convert-usd`` buys the exact ``USDCUSD`` shortfall on Kraken; by
     default, refuses shortfall and recommends 0%-fee Kraken UI Convert (saves 20 bps).
  6. Polls Kraken ``WithdrawStatus`` + Arbitrum One on-chain USDC arrival, prints the
     2-click Hyperliquid Deposit + Perp->Spot steps, and optionally watches Hyperliquid
     Spot arrival (``--watch-hl``) and sends ``ntfy`` notifications (``--notify``).

Examples:
  # Read-only status across Kraken, Arbitrum One, and Hyperliquid
  ./myenv/bin/python verify_tools/rebalance_kraken_to_hl.py

  # Dry-run a small 11 USDC test transfer (10 USDC net after 1 USDC Arbitrum fee)
  ./myenv/bin/python verify_tools/rebalance_kraken_to_hl.py --amount 11

  # Execute a small 11 USDC test transfer (requires USDC on Kraken, or add --convert-usd)
  ./myenv/bin/python verify_tools/rebalance_kraken_to_hl.py --amount 11 --execute --notify

  # Dry-run moving 50% of Kraken USD+USDC cash to Hyperliquid
  ./myenv/bin/python verify_tools/rebalance_kraken_to_hl.py --fraction 0.5
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import (  # noqa: E402
    clean_scalar_str,
    clean_upper_scalar,
    load_env_stack,
    optional_env,
    parse_dotenv,
    required_float_env,
    safe_nonneg_float,
    safe_pos_float,
)
from core.http import fetch_json_post  # noqa: E402
from credentials import kraken_credentials  # noqa: E402
from notify_engine import alertnotifiers as alert  # noqa: E402
from providers.kraken_provider import KrakenProvider  # noqa: E402
from state_io import encode_json_payload  # noqa: E402

_EVM_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ERC20_BALANCE_OF_SELECTOR = "0x70a08231"
_USDC_DECIMALS_DIVISOR = 1_000_000.0
_ETH_DECIMALS_DIVISOR = 1_000_000_000_000_000_000.0
_DEFAULT_FEE_CUSHION_USDC = 0.25
_USDCUSD_PAIR = "USDCUSD"
_USDCUSD_MIN_ORDER = 5.0
_CONVERT_POLL_SEC = 2.0
_CONVERT_TIMEOUT_SEC = 90.0


@dataclass(frozen=True)
class RebalanceConfig:
    withdraw_asset: str
    withdraw_method: str
    arb_rpc_url: str
    hl_info_url: str
    arb_usdc_contract: str
    min_net_usdc: float
    min_arb_eth_gas: float
    poll_interval_sec: float
    poll_timeout_sec: float
    target_address: str
    configured_withdraw_key: str | None


def _require_nonempty_cfg(cfg: dict[str, str], key: str) -> str:
    val = clean_scalar_str(optional_env(key) or cfg.get(key))
    if not val:
        raise ValueError(f"Missing required Kraken transfer config key: {key}")
    return val


def _require_pos_cfg(cfg: dict[str, str], key: str) -> float:
    raw = optional_env(key) or cfg.get(key)
    val = required_float_env(key, {key: raw})
    if val <= 0.0:
        raise ValueError(f"Config key {key} must be positive, got {val!r}")
    return val


def validate_evm_address(address: object, label: str = "address") -> str:
    addr = clean_scalar_str(address)
    if not addr or not _EVM_ADDR_RE.fullmatch(addr):
        raise ValueError(f"Invalid EVM {label}: {address!r}")
    return addr


def load_rebalance_config(
    *,
    root_dir: str = ROOT,
    to_address_override: str | None = None,
    withdraw_key_override: str | None = None,
) -> RebalanceConfig:
    """Load non-secret transfer policy from kraken/config.env and wallet address from hyperliquid/.env."""
    kraken_cfg_path = os.path.join(root_dir, "kraken", "config.env")
    kraken_env_path = os.path.join(root_dir, "kraken", ".env")
    hl_env_path = os.path.join(root_dir, "hyperliquid", ".env")

    load_env_stack(kraken_env_path)
    cfg = parse_dotenv(kraken_cfg_path)
    hl_env = parse_dotenv(hl_env_path) if os.path.isfile(hl_env_path) else {}

    raw_target = to_address_override or optional_env("HL_ACCOUNT_ADDRESS") or hl_env.get("HL_ACCOUNT_ADDRESS")
    target_address = validate_evm_address(raw_target, label="Hyperliquid target address")
    arb_usdc_contract = validate_evm_address(
        _require_nonempty_cfg(cfg, "KRAKEN_HL_ARB_USDC_CONTRACT"),
        label="Arbitrum native USDC contract",
    )
    raw_key = clean_scalar_str(withdraw_key_override or optional_env("KRAKEN_HL_ARB_WITHDRAW_KEY"))

    return RebalanceConfig(
        withdraw_asset=_require_nonempty_cfg(cfg, "KRAKEN_HL_WITHDRAW_ASSET").upper(),
        withdraw_method=_require_nonempty_cfg(cfg, "KRAKEN_HL_WITHDRAW_METHOD"),
        arb_rpc_url=_require_nonempty_cfg(cfg, "KRAKEN_HL_ARB_RPC_URL"),
        hl_info_url=_require_nonempty_cfg(cfg, "KRAKEN_HL_INFO_URL"),
        arb_usdc_contract=arb_usdc_contract,
        min_net_usdc=_require_pos_cfg(cfg, "KRAKEN_HL_MIN_NET_USDC"),
        min_arb_eth_gas=_require_pos_cfg(cfg, "KRAKEN_HL_MIN_ARB_ETH_GAS"),
        poll_interval_sec=_require_pos_cfg(cfg, "KRAKEN_HL_POLL_INTERVAL_SEC"),
        poll_timeout_sec=_require_pos_cfg(cfg, "KRAKEN_HL_POLL_TIMEOUT_SEC"),
        target_address=target_address,
        configured_withdraw_key=raw_key,
    )


def _hex_to_nonneg_int(raw: object) -> int:
    s = clean_scalar_str(raw)
    if not s:
        raise ValueError(f"Empty hex RPC response: {raw!r}")
    body = s[2:] if s.lower().startswith("0x") else s
    if not body:
        return 0
    if len(body) % 2 == 1:
        body = "0" + body
    return int.from_bytes(bytes.fromhex(body), "big")


def fetch_arbitrum_balances(cfg: RebalanceConfig, address: str) -> dict[str, float]:
    """Read Arbitrum One ETH (gas) and native USDC balances via JSON-RPC."""
    addr = validate_evm_address(address, label="Arbitrum address")
    rpc_headers = {"User-Agent": "mptrade-rebalance/1.0"}

    eth_resp = fetch_json_post(
        cfg.arb_rpc_url,
        {"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance", "params": [addr, "latest"]},
        timeout=10.0,
        headers=rpc_headers,
    )
    if not isinstance(eth_resp, dict) or "result" not in eth_resp:
        raise RuntimeError(f"Arbitrum eth_getBalance failed: {eth_resp!r}")
    eth_bal = _hex_to_nonneg_int(eth_resp["result"]) / _ETH_DECIMALS_DIVISOR

    call_data = _ERC20_BALANCE_OF_SELECTOR + addr[2:].lower().rjust(64, "0")
    usdc_resp = fetch_json_post(
        cfg.arb_rpc_url,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_call",
            "params": [{"to": cfg.arb_usdc_contract, "data": call_data}, "latest"],
        },
        timeout=10.0,
        headers=rpc_headers,
    )
    if not isinstance(usdc_resp, dict) or "result" not in usdc_resp:
        raise RuntimeError(f"Arbitrum USDC balanceOf failed: {usdc_resp!r}")
    usdc_bal = _hex_to_nonneg_int(usdc_resp["result"]) / _USDC_DECIMALS_DIVISOR

    return {"eth": round(eth_bal, 8), "usdc": round(usdc_bal, 6)}


def fetch_hyperliquid_balances(cfg: RebalanceConfig, address: str) -> dict[str, float]:
    """Read Hyperliquid Spot (USDC, HYPE) and Perp account equity via public Info API."""
    addr = validate_evm_address(address, label="Hyperliquid address")
    spot_resp = fetch_json_post(
        cfg.hl_info_url,
        {"type": "spotClearinghouseState", "user": addr},
        timeout=10.0,
    )
    perp_resp = fetch_json_post(
        cfg.hl_info_url,
        {"type": "clearinghouseState", "user": addr},
        timeout=10.0,
    )
    spot_usdc = 0.0
    spot_hype = 0.0
    if isinstance(spot_resp, dict):
        for row in spot_resp.get("balances") or ():
            if not isinstance(row, dict):
                continue
            coin = clean_upper_scalar(row.get("coin"))
            total = safe_nonneg_float(row.get("total"), default=0.0) or 0.0
            if coin == "USDC":
                spot_usdc = total
            elif coin == "HYPE":
                spot_hype = total
    perp_equity = 0.0
    perp_withdrawable = 0.0
    if isinstance(perp_resp, dict):
        ms = perp_resp.get("marginSummary") if isinstance(perp_resp.get("marginSummary"), dict) else {}
        perp_equity = safe_nonneg_float(ms.get("accountValue"), default=0.0) or 0.0
        perp_withdrawable = safe_nonneg_float(perp_resp.get("withdrawable"), default=0.0) or 0.0

    return {
        "spot_usdc": round(spot_usdc, 6),
        "spot_hype": round(spot_hype, 6),
        "perp_equity_usd": round(perp_equity, 6),
        "perp_withdrawable_usd": round(perp_withdrawable, 6),
    }


def extract_kraken_cash(balance_map: dict[str, Any]) -> dict[str, float]:
    """Extract liquid USD (ZUSD) and USDC balances from Kraken Balance response."""
    zusd = safe_nonneg_float(balance_map.get("ZUSD"), default=0.0) or 0.0
    usd_alt = safe_nonneg_float(balance_map.get("USD"), default=0.0) or 0.0
    usdc = safe_nonneg_float(balance_map.get("USDC"), default=0.0) or 0.0
    usd_total = zusd if zusd > 0.0 else usd_alt
    return {
        "zusd": round(usd_total, 4),
        "usdc": round(usdc, 6),
        "total_usd_equiv": round(usd_total + usdc, 4),
    }


def select_native_arbitrum_method(methods: list[dict[str, Any]], cfg: RebalanceConfig) -> dict[str, Any] | None:
    """Return the exact Kraken WithdrawMethods row for Native Arbitrum One USDC."""
    wanted_asset = cfg.withdraw_asset.upper()
    wanted_method = cfg.withdraw_method.strip().lower()
    for row in methods:
        if not isinstance(row, dict):
            continue
        asset_u = clean_upper_scalar(row.get("asset")) or ""
        method_s = (clean_scalar_str(row.get("method")) or "").lower()
        if asset_u == wanted_asset and method_s == wanted_method:
            return row
    return None


def resolve_verified_withdraw_address(
    addresses: list[dict[str, Any]],
    cfg: RebalanceConfig,
    *,
    requested_key: str | None = None,
) -> dict[str, Any]:
    """Fail-closed resolution of a verified Kraken whitelisted address for Native Arbitrum USDC.

    Explicitly rejects bridged ``Arbitrum One Network`` (USDC.e), unverified entries,
    and any address not matching ``cfg.target_address``.
    """
    wanted_asset = cfg.withdraw_asset.upper()
    wanted_method = cfg.withdraw_method.strip().lower()
    wanted_addr = cfg.target_address.lower()
    key_filter = clean_scalar_str(requested_key or cfg.configured_withdraw_key)

    exact_matches: list[dict[str, Any]] = []
    near_mismatches: list[str] = []

    for row in addresses:
        if not isinstance(row, dict):
            continue
        r_asset = clean_upper_scalar(row.get("asset")) or ""
        r_method = clean_scalar_str(row.get("method")) or ""
        r_addr = clean_scalar_str(row.get("address")) or ""
        r_key = clean_scalar_str(row.get("key")) or ""
        r_verified = bool(row.get("verified") is True)

        if key_filter and r_key != key_filter:
            continue
        if r_asset != wanted_asset:
            if key_filter:
                near_mismatches.append(f"key={r_key!r} has asset={r_asset!r} (expected {wanted_asset!r})")
            continue
        if r_method.lower() != wanted_method:
            near_mismatches.append(
                f"key={r_key!r} uses method={r_method!r} (REJECTED: expected exact {cfg.withdraw_method!r}; "
                f"never use bridged USDC.e for Hyperliquid)"
            )
            continue
        if r_addr.lower() != wanted_addr:
            near_mismatches.append(
                f"key={r_key!r} points to {r_addr!r} (expected Hyperliquid wallet {cfg.target_address!r})"
            )
            continue
        if not r_verified:
            near_mismatches.append(f"key={r_key!r} is NOT email-verified yet on Kraken")
            continue
        exact_matches.append(row)

    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        keys_str = ", ".join(repr(m.get("key")) for m in exact_matches)
        raise ValueError(
            f"Multiple verified {cfg.withdraw_method!r} keys match {cfg.target_address}: {keys_str}. "
            f"Pass --withdraw-key <name> or set KRAKEN_HL_ARB_WITHDRAW_KEY in kraken/.env."
        )
    detail = "; ".join(near_mismatches) if near_mismatches else "no USDC Arbitrum One Native address whitelisted yet"
    raise ValueError(
        f"No verified Kraken withdrawal address matches {cfg.target_address} on {cfg.withdraw_method!r} ({detail}). "
        f"In Kraken UI -> Security -> Withdrawal Addresses, add {cfg.target_address} under asset USDC and network "
        f"'{cfg.withdraw_method}' (Arbitrum One Native USDC, NOT USDC.e), confirm the email, then re-run."
    )


def convert_usd_to_usdc_on_kraken(
    client: Any,
    needed_usdc: float,
    *,
    market_order: bool = False,
    poll_sec: float = _CONVERT_POLL_SEC,
    timeout_sec: float = _CONVERT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Convert ZUSD -> USDC via Kraken Spot USDCUSD (20 bps fee) and wait for fill."""
    qty = round(max(_USDCUSD_MIN_ORDER, needed_usdc), 4)
    tk = client.ticker(_USDCUSD_PAIR)
    if not isinstance(tk, dict):
        raise RuntimeError("Unable to fetch Kraken USDCUSD ticker")
    ask = safe_pos_float((tk.get("a") or [None])[0], default=None)
    bid = safe_pos_float((tk.get("b") or [None])[0], default=None)
    if ask is None or bid is None:
        raise RuntimeError(f"Invalid Kraken USDCUSD BBO: {tk!r}")

    if market_order:
        res = client.add_order(_USDCUSD_PAIR, "buy", qty, ordertype="market", post_only=False)
    else:
        # Cross at ask (or sit at bid if post_only) -- for immediate conversion without resting, use limit at ask.
        res = client.add_order(_USDCUSD_PAIR, "buy", qty, price=ask, ordertype="limit", post_only=False)

    txids = res.get("txid") if isinstance(res, dict) else None
    txid = clean_scalar_str(txids[0]) if isinstance(txids, list) and txids else None
    if not txid:
        raise RuntimeError(f"Kraken USDCUSD order did not return a txid: {res!r}")

    deadline = time.monotonic() + timeout_sec
    last_status = "open"
    filled_vol = 0.0
    fee_usd = 0.0
    while time.monotonic() <= deadline:
        q = client.query_orders(txid)
        info = q.get(txid) if isinstance(q, dict) else None
        if isinstance(info, dict):
            last_status = clean_scalar_str(info.get("status")) or last_status
            filled_vol = safe_nonneg_float(info.get("vol_exec"), default=0.0) or 0.0
            fee_usd = safe_nonneg_float(info.get("fee"), default=0.0) or 0.0
            if last_status == "closed":
                return {
                    "txid": txid,
                    "status": last_status,
                    "requested_usdc": qty,
                    "filled_usdc": filled_vol,
                    "fee_usd": fee_usd,
                }
            if last_status in {"canceled", "expired"}:
                raise RuntimeError(f"Kraken USDCUSD order {txid} ended in status={last_status}")
        time.sleep(poll_sec)
    raise RuntimeError(
        f"Timed out waiting for Kraken USDCUSD order {txid} to fill (status={last_status}, filled={filled_vol})"
    )


def wait_for_arbitrum_arrival(
    client: Any,
    cfg: RebalanceConfig,
    refid: str,
    baseline_arb_usdc: float,
    expected_net_usdc: float,
) -> dict[str, Any]:
    """Poll Kraken WithdrawStatus and Arbitrum One USDC balance until arrival."""
    deadline = time.monotonic() + cfg.poll_timeout_sec
    target_arb_usdc = baseline_arb_usdc + max(0.0, expected_net_usdc - 0.01)
    last_k_status = "Unknown"
    chain_txid: str | None = None

    while time.monotonic() <= deadline:
        try:
            rows = client.withdraw_status(cfg.withdraw_asset)
            for row in rows:
                if clean_scalar_str(row.get("refid")) == refid:
                    last_k_status = clean_scalar_str(row.get("status")) or last_k_status
                    chain_txid = clean_scalar_str(row.get("txid")) or chain_txid
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] WithdrawStatus poll failed: {exc}", flush=True)

        arb = fetch_arbitrum_balances(cfg, cfg.target_address)
        delta = arb["usdc"] - baseline_arb_usdc
        print(
            f"[WAIT-ARB] refid={refid} kraken_status={last_k_status} txid={chain_txid or '-'} "
            f"arb_usdc={arb['usdc']:.4f} (+{delta:.4f} / expected {expected_net_usdc:.4f})",
            flush=True,
        )
        if last_k_status.lower() in {"canceled", "failure", "failed"}:
            raise RuntimeError(f"Kraken withdrawal {refid} failed with status={last_k_status}")
        if arb["usdc"] >= target_arb_usdc:
            return {
                "arrived": True,
                "refid": refid,
                "kraken_status": last_k_status,
                "arb_txid": chain_txid,
                "arb_usdc_before": baseline_arb_usdc,
                "arb_usdc_after": arb["usdc"],
                "arb_usdc_delta": round(delta, 6),
            }
        time.sleep(cfg.poll_interval_sec)

    return {
        "arrived": False,
        "refid": refid,
        "kraken_status": last_k_status,
        "arb_txid": chain_txid,
        "arb_usdc_before": baseline_arb_usdc,
        "arb_usdc_after": fetch_arbitrum_balances(cfg, cfg.target_address)["usdc"],
    }


def watch_hyperliquid_spot_arrival(
    cfg: RebalanceConfig,
    baseline_spot_usdc: float,
    expected_net_usdc: float,
) -> dict[str, Any]:
    """Poll Hyperliquid Spot & Perp balances until deposited funds land in Spot USDC."""
    deadline = time.monotonic() + cfg.poll_timeout_sec
    target_spot = baseline_spot_usdc + max(0.0, expected_net_usdc - 0.05)

    while time.monotonic() <= deadline:
        hl = fetch_hyperliquid_balances(cfg, cfg.target_address)
        delta_spot = hl["spot_usdc"] - baseline_spot_usdc
        print(
            f"[WAIT-HL] spot_usdc={hl['spot_usdc']:.4f} (+{delta_spot:.4f}) "
            f"perp_usd={hl['perp_equity_usd']:.4f} (need spot +{expected_net_usdc:.4f})",
            flush=True,
        )
        if hl["spot_usdc"] >= target_spot:
            return {"arrived_hl_spot": True, **hl, "spot_usdc_delta": round(delta_spot, 6)}
        time.sleep(cfg.poll_interval_sec)

    hl = fetch_hyperliquid_balances(cfg, cfg.target_address)
    return {"arrived_hl_spot": False, **hl}


def build_status_snapshot(client: Any, cfg: RebalanceConfig, *, requested_key: str | None = None) -> dict[str, Any]:
    """Collect a read-only snapshot across Kraken, Arbitrum One, and Hyperliquid."""
    cash = extract_kraken_cash(client.balance())
    methods = client.withdraw_methods(cfg.withdraw_asset)
    method_row = select_native_arbitrum_method(methods, cfg)
    addresses = client.withdraw_addresses(cfg.withdraw_asset)
    arb = fetch_arbitrum_balances(cfg, cfg.target_address)
    hl = fetch_hyperliquid_balances(cfg, cfg.target_address)

    resolved_addr: dict[str, Any] | None = None
    addr_error: str | None = None
    try:
        resolved_addr = resolve_verified_withdraw_address(addresses, cfg, requested_key=requested_key)
    except ValueError as exc:
        addr_error = str(exc)

    method_fee = None
    method_min = None
    if isinstance(method_row, dict):
        fee_obj = method_row.get("fee")
        if isinstance(fee_obj, dict):
            method_fee = safe_nonneg_float(fee_obj.get("fee"), default=None)
        method_min = safe_pos_float(method_row.get("minimum"), default=None)

    return {
        "target_address": cfg.target_address,
        "withdraw_asset": cfg.withdraw_asset,
        "withdraw_method": cfg.withdraw_method,
        "kraken_cash": cash,
        "kraken_method": {
            "found": method_row is not None,
            "fee_usdc": method_fee,
            "minimum_usdc": method_min,
            "network": method_row.get("network") if isinstance(method_row, dict) else None,
        },
        "kraken_whitelisted_address": {
            "ready": resolved_addr is not None,
            "key": resolved_addr.get("key") if resolved_addr else None,
            "address": resolved_addr.get("address") if resolved_addr else None,
            "method": resolved_addr.get("method") if resolved_addr else None,
            "verified": resolved_addr.get("verified") if resolved_addr else False,
            "error": addr_error,
        },
        "arbitrum_one": {
            **arb,
            "gas_ok": arb["eth"] >= cfg.min_arb_eth_gas,
            "min_eth_gas": cfg.min_arb_eth_gas,
        },
        "hyperliquid": hl,
    }


def plan_and_run_transfer(
    client: Any,
    cfg: RebalanceConfig,
    *,
    amount_usdc: float | None = None,
    fraction: float | None = None,
    requested_key: str | None = None,
    convert_usd: bool = False,
    market_convert: bool = False,
    max_fee_override: float | None = None,
    execute: bool = False,
    wait_arb: bool = True,
    watch_hl: bool = False,
    send_notify: bool = False,
) -> dict[str, Any]:
    """Validate, dry-run, or execute a Variant-A Kraken -> Arbitrum One USDC transfer."""
    snapshot = build_status_snapshot(client, cfg, requested_key=requested_key)
    if amount_usdc is None and fraction is None:
        return {"mode": "status", "snapshot": snapshot}

    if amount_usdc is not None and fraction is not None:
        raise ValueError("Specify either --amount or --fraction, not both")

    cash = snapshot["kraken_cash"]
    if fraction is not None:
        frac_f = safe_pos_float(fraction, default=None)
        if frac_f is None or frac_f > 1.0:
            raise ValueError(f"--fraction must be in (0, 1], got {fraction!r}")
        gross_usdc = round(cash["total_usd_equiv"] * frac_f, 2)
    else:
        gross_f = safe_pos_float(amount_usdc, default=None)
        if gross_f is None:
            raise ValueError(f"--amount must be positive, got {amount_usdc!r}")
        gross_usdc = round(gross_f, 4)

    if not snapshot["kraken_whitelisted_address"]["ready"]:
        raise ValueError(snapshot["kraken_whitelisted_address"]["error"])

    withdraw_key = str(snapshot["kraken_whitelisted_address"]["key"])
    if not snapshot["arbitrum_one"]["gas_ok"]:
        raise ValueError(
            f"Arbitrum One wallet {cfg.target_address} has only {snapshot['arbitrum_one']['eth']:.6f} ETH "
            f"(minimum required gas: {cfg.min_arb_eth_gas} ETH)"
        )

    winfo = client.withdraw_info(cfg.withdraw_asset, withdraw_key, gross_usdc)
    fee_usdc = safe_nonneg_float(winfo.get("fee"), default=None)
    net_usdc = safe_nonneg_float(winfo.get("amount"), default=None)
    limit_usdc = safe_nonneg_float(winfo.get("limit"), default=None)
    if fee_usdc is None or net_usdc is None:
        raise RuntimeError(f"Unexpected Kraken WithdrawInfo response: {winfo!r}")

    if net_usdc < cfg.min_net_usdc:
        raise ValueError(
            f"Net USDC receipt ({net_usdc:.4f} USDC = {gross_usdc:.4f} - {fee_usdc:.4f} fee) is below "
            f"KRAKEN_HL_MIN_NET_USDC={cfg.min_net_usdc:.2f} (Hyperliquid Bridge2 permanently loses deposits < 5 USDC)."
        )
    if limit_usdc is not None and gross_usdc > limit_usdc:
        raise ValueError(f"Requested {gross_usdc:.4f} USDC exceeds Kraken remaining withdrawal limit {limit_usdc:.4f}")

    shortfall_usdc = round(max(0.0, gross_usdc - cash["usdc"]), 4)
    est_convert_fee_usd = round(shortfall_usdc * 0.002, 4) if shortfall_usdc > 0.0 else 0.0
    max_fee = (
        safe_pos_float(max_fee_override, default=None)
        if max_fee_override is not None
        else round(fee_usdc + _DEFAULT_FEE_CUSHION_USDC, 4)
    )
    if max_fee is None:
        raise ValueError(f"Invalid --max-fee: {max_fee_override!r}")

    plan = {
        "gross_usdc": gross_usdc,
        "fee_usdc": fee_usdc,
        "net_arbitrum_usdc": net_usdc,
        "max_fee_usdc": max_fee,
        "withdraw_key": withdraw_key,
        "withdraw_method": cfg.withdraw_method,
        "target_address": cfg.target_address,
        "kraken_usdc_available": cash["usdc"],
        "kraken_zusd_available": cash["zusd"],
        "usdc_shortfall": shortfall_usdc,
        "convert_usd_enabled": convert_usd,
        "est_spot_convert_fee_usd": est_convert_fee_usd,
    }

    if shortfall_usdc > 0.0:
        needed_zusd = shortfall_usdc * 1.003
        if cash["zusd"] < needed_zusd:
            raise ValueError(
                f"Insufficient Kraken cash: need {shortfall_usdc:.4f} more USDC (~${needed_zusd:.2f} ZUSD), "
                f"available ZUSD=${cash['zusd']:.2f}."
            )
        if not convert_usd and execute:
            raise ValueError(
                f"Kraken has {cash['usdc']:.4f} USDC vs {gross_usdc:.4f} requested (shortfall {shortfall_usdc:.4f} USDC). "
                f"Convert USD -> USDC 1:1 with 0% fee in Kraken UI Convert (saves ~${est_convert_fee_usd:.2f}), "
                f"or re-run with --convert-usd to buy on Kraken Spot USDCUSD (~20 bps fee)."
            )

    if not execute:
        return {"mode": "dry_run", "plan": plan, "snapshot": snapshot}

    conversion_result: dict[str, Any] | None = None
    if shortfall_usdc > 0.0 and convert_usd:
        # Include 0.25% buffer for Kraken Spot USDCUSD fee so post-fee USDC covers gross_usdc.
        order_usdc = round(max(_USDCUSD_MIN_ORDER, shortfall_usdc * 1.0025), 4)
        conversion_result = convert_usd_to_usdc_on_kraken(
            client, order_usdc, market_order=market_convert
        )
        post_cash = extract_kraken_cash(client.balance())
        if post_cash["usdc"] + 1e-6 < gross_usdc:
            raise RuntimeError(
                f"Post-conversion USDC balance ({post_cash['usdc']:.4f}) still below gross_usdc ({gross_usdc:.4f})"
            )

    baseline_arb_usdc = snapshot["arbitrum_one"]["usdc"]
    baseline_hl_spot_usdc = snapshot["hyperliquid"]["spot_usdc"]

    wres = client.withdraw(
        cfg.withdraw_asset,
        withdraw_key,
        gross_usdc,
        address=cfg.target_address,
        max_fee=max_fee,
    )
    refid = clean_scalar_str(wres.get("refid"))
    if not refid:
        raise RuntimeError(f"Kraken Withdraw did not return a refid: {wres!r}")

    out: dict[str, Any] = {
        "mode": "executed",
        "plan": plan,
        "conversion": conversion_result,
        "withdraw_response": wres,
        "refid": refid,
        "next_steps": [
            f"1. Open https://app.hyperliquid.xyz connected with wallet {cfg.target_address}",
            f"2. Click Deposit -> Network: Arbitrum -> Native USDC -> Amount: ~{net_usdc:.2f} USDC (sign in wallet)",
            "3. Click Transfer -> Perp to Spot -> move deposited USDC to Spot for HYPEUSDC trading",
        ],
    }

    if wait_arb:
        arb_res = wait_for_arbitrum_arrival(client, cfg, refid, baseline_arb_usdc, net_usdc)
        out["arbitrum_arrival"] = arb_res
        if send_notify and arb_res.get("arrived"):
            alert.notify(
                f"Kraken -> Arbitrum USDC arrived (+{arb_res['arb_usdc_delta']:.2f} USDC)",
                (
                    f"Wallet {cfg.target_address} received +{arb_res['arb_usdc_delta']:.2f} native USDC on Arbitrum One "
                    f"(txid={arb_res.get('arb_txid') or 'pending'}).\n"
                    f"Next: app.hyperliquid.xyz -> Deposit (Arbitrum USDC) -> Transfer (Perp to Spot)."
                ),
                tags="moneybag",
            )
    if watch_hl:
        hl_res = watch_hyperliquid_spot_arrival(cfg, baseline_hl_spot_usdc, net_usdc)
        out["hyperliquid_arrival"] = hl_res
        if send_notify and hl_res.get("arrived_hl_spot"):
            alert.notify(
                f"Hyperliquid Spot USDC credited (+{hl_res['spot_usdc_delta']:.2f} USDC)",
                f"Hyperliquid Spot USDC balance is now {hl_res['spot_usdc']:.2f} USDC on {cfg.target_address}.",
                tags="white_check_mark",
            )

    return out


def _print_human_report(result: dict[str, Any]) -> None:
    mode = result.get("mode", "status")
    snap = result.get("snapshot")
    if isinstance(snap, dict):
        kc = snap["kraken_cash"]
        km = snap["kraken_method"]
        ka = snap["kraken_whitelisted_address"]
        arb = snap["arbitrum_one"]
        hl = snap["hyperliquid"]
        print("=== Kraken -> Hyperliquid Rebalance Status (Variant A) ===")
        print(f"  Target wallet       : {snap['target_address']}")
        print(f"  Kraken cash         : ZUSD=${kc['zusd']:.2f} | USDC={kc['usdc']:.4f} | Total=${kc['total_usd_equiv']:.2f}")
        print(
            f"  Kraken network      : {snap['withdraw_method']} "
            f"(fee={km['fee_usdc']} USDC, min={km['minimum_usdc']} USDC, network={km['network']})"
        )
        if ka["ready"]:
            print(f"  Whitelisted address : READY key={ka['key']!r} -> {ka['address']} (verified=True)")
        else:
            print(f"  Whitelisted address : NOT READY -- {ka['error']}")
        print(
            f"  Arbitrum One wallet : ETH={arb['eth']:.6f} ({'gas OK' if arb['gas_ok'] else 'LOW GAS'}) | "
            f"Native USDC={arb['usdc']:.4f}"
        )
        print(
            f"  Hyperliquid L1      : Spot USDC={hl['spot_usdc']:.4f} | Spot HYPE={hl['spot_hype']:.4f} | "
            f"Perp USD={hl['perp_equity_usd']:.4f}"
        )

    plan = result.get("plan")
    if isinstance(plan, dict):
        tag = "DRY-RUN PLAN (pass --execute to send)" if mode == "dry_run" else "EXECUTED TRANSFER"
        print(f"\n=== {tag} ===")
        print(
            f"  Gross withdraw      : {plan['gross_usdc']:.4f} USDC -> fee={plan['fee_usdc']:.4f} USDC "
            f"(max_fee={plan['max_fee_usdc']:.4f}) -> Net Arbitrum={plan['net_arbitrum_usdc']:.4f} USDC"
        )
        print(f"  Destination key     : {plan['withdraw_key']!r} ({plan['withdraw_method']}) -> {plan['target_address']}")
        if plan["usdc_shortfall"] > 0.0:
            if plan["convert_usd_enabled"]:
                print(
                    f"  USD -> USDC convert : shortfall={plan['usdc_shortfall']:.4f} USDC via Kraken Spot USDCUSD "
                    f"(est. 20 bps fee ~${plan['est_spot_convert_fee_usd']:.2f})"
                )
            else:
                print(
                    f"  [ACTION NEEDED]     : shortfall={plan['usdc_shortfall']:.4f} USDC. Convert USD -> USDC 1:1 "
                    f"with 0% fee in Kraken UI Convert (saves ~${plan['est_spot_convert_fee_usd']:.2f}), "
                    f"or pass --convert-usd."
                )

    if mode == "executed":
        print(f"  Kraken refid        : {result.get('refid')}")
        if result.get("conversion"):
            print(f"  Spot conversion     : {result['conversion']}")
        if result.get("arbitrum_arrival"):
            print(f"  Arbitrum arrival    : {result['arbitrum_arrival']}")
        if result.get("hyperliquid_arrival"):
            print(f"  Hyperliquid arrival : {result['hyperliquid_arrival']}")
        print("\n=== Next Steps (2 clicks in Hyperliquid UI, signed by your wallet) ===")
        for step in result.get("next_steps") or ():
            print(f"  {step}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Variant-A capital transfer tool: Kraken -> Arbitrum One -> Hyperliquid Spot (dry-run by default)."
    )
    p.add_argument("--amount", type=float, default=None, help="Gross USDC amount to withdraw from Kraken.")
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of total Kraken USD+USDC cash to move (e.g. 0.5 for half).",
    )
    p.add_argument("--withdraw-key", type=str, default=None, help="Kraken whitelisted withdrawal key label.")
    p.add_argument("--to-address", type=str, default=None, help="Override target EVM address (defaults to HL_ACCOUNT_ADDRESS).")
    p.add_argument("--max-fee", type=float, default=None, help="Maximum Kraken withdrawal fee cap in USDC.")
    p.add_argument(
        "--convert-usd",
        action="store_true",
        help="Buy exact USDC shortfall on Kraken Spot USDCUSD (~20 bps fee) before withdrawing.",
    )
    p.add_argument(
        "--market-convert",
        action="store_true",
        help="Use market order instead of ask-limit when --convert-usd is enabled.",
    )
    p.add_argument("--execute", action="store_true", help="Execute live transfer (omit for safe dry-run).")
    p.add_argument("--no-wait", action="store_true", help="Do not wait for Arbitrum One USDC arrival after withdrawal.")
    p.add_argument(
        "--watch-hl",
        action="store_true",
        help="Also poll Hyperliquid Spot USDC until your manual Deposit + Perp->Spot transfer completes.",
    )
    p.add_argument("--notify", action="store_true", help="Send ntfy notification when funds arrive on Arbitrum / HL.")
    p.add_argument("--json", action="store_true", help="Emit JSON output instead of human-readable report.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_rebalance_config(
            to_address_override=args.to_address,
            withdraw_key_override=args.withdraw_key,
        )
        creds = kraken_credentials("transfer")
        client = KrakenProvider.build_client(creds.key, creds.secret)
        res = plan_and_run_transfer(
            client,
            cfg,
            amount_usdc=args.amount,
            fraction=args.fraction,
            requested_key=args.withdraw_key,
            convert_usd=bool(args.convert_usd),
            market_convert=bool(args.market_convert),
            max_fee_override=args.max_fee,
            execute=bool(args.execute),
            wait_arb=not bool(args.no_wait),
            watch_hl=bool(args.watch_hl),
            send_notify=bool(args.notify),
        )
    except Exception as exc:  # noqa: BLE001
        if args.json:
            print(encode_json_payload({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(encode_json_payload({"ok": True, **res}, indent=2))
    else:
        _print_human_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
