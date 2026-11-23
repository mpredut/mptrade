"""Tests for Kraken transfer credentials, KrakenClient withdrawal endpoints, and rebalance_kraken_to_hl."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import credentials
from kraken.kraken_client import KrakenClient, _WRITE_METHODS
from verify_tools import rebalance_kraken_to_hl as reb

_WALLET = "0xBf48fd0c624c1ab06FDc87F919C0A9108bD90233"
_OTHER_WALLET = "0x1111111111111111111111111111111111111111"


def _make_cfg(**overrides) -> reb.RebalanceConfig:
    base = dict(
        withdraw_asset="USDC",
        withdraw_method="Arbitrum One Network - Native",
        arb_rpc_url="https://arb1.arbitrum.io/rpc",
        hl_info_url="https://api.hyperliquid.xyz/info",
        arb_usdc_contract="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        min_net_usdc=6.0,
        min_arb_eth_gas=0.0001,
        poll_interval_sec=0.01,
        poll_timeout_sec=1.0,
        target_address=_WALLET,
        configured_withdraw_key=None,
    )
    base.update(overrides)
    return reb.RebalanceConfig(**base)


def test_kraken_transfer_credential_profile_is_isolated_from_bot():
    pair = credentials.kraken_credentials(
        "transfer",
        values={
            "KRAKEN_API_KEY_TRANSFER": "tk",
            "KRAKEN_API_SECRET_TRANSFER": "ts",
            "KRAKEN_API_KEY_BOT": "bk",
            "KRAKEN_API_SECRET_BOT": "bs",
        },
    )
    assert pair.key == "tk"
    assert pair.secret == "ts"
    assert pair.profile == "Kraken TRANSFER"

    # Never fall back to BOT keys when TRANSFER keys are missing.
    with pytest.raises(credentials.CredentialProfileMissingError):
        credentials.kraken_credentials(
            "transfer",
            values={"KRAKEN_API_KEY_BOT": "bk", "KRAKEN_API_SECRET_BOT": "bs"},
        )

    # Partial TRANSFER profile fails fast.
    with pytest.raises(credentials.CredentialConfigurationError):
        credentials.kraken_credentials("transfer", values={"KRAKEN_API_KEY_TRANSFER": "only_key"})


def test_kraken_client_withdraw_write_guard_and_list_parsing():
    assert "Withdraw" in _WRITE_METHODS
    assert "WithdrawCancel" in _WRITE_METHODS

    cli = KrakenClient("key", "c2VjcmV0")
    list_body = b'{"error": [], "result": [{"method": "Arbitrum One Network - Native", "asset": "USDC"}]}'
    assert cli._parse(200, list_body, allow_list=False) == {}
    assert cli._parse(200, list_body, allow_list=True) == [
        {"method": "Arbitrum One Network - Native", "asset": "USDC"}
    ]

    with patch("kraken.kraken_client.http_post_form", return_value=(200, list_body)) as mock_post:
        methods = cli.withdraw_methods("usdc")
        assert len(methods) == 1
        assert methods[0]["method"] == "Arbitrum One Network - Native"
        assert mock_post.call_count == 1

    # Withdraw must NEVER auto-retry on transient network failure (max_attempts == 1).
    with patch("kraken.kraken_client.http_post_form", return_value=(0, b"")) as mock_post:
        with pytest.raises(Exception):
            cli.withdraw("USDC", "HL_ARB_MAIN", 11.0, address=_WALLET, max_fee=1.25)
        assert mock_post.call_count == 1


def test_resolve_verified_withdraw_address_rejects_usdce_and_mismatches():
    cfg = _make_cfg()

    # Rejects bridged USDC.e ("Arbitrum One Network" without " - Native")
    with pytest.raises(ValueError, match="USDC.e"):
        reb.resolve_verified_withdraw_address(
            [
                {
                    "asset": "USDC",
                    "method": "Arbitrum One Network",
                    "address": _WALLET,
                    "key": "wrong_bridged",
                    "verified": True,
                }
            ],
            cfg,
        )

    # Rejects unverified address
    with pytest.raises(ValueError, match="NOT email-verified"):
        reb.resolve_verified_withdraw_address(
            [
                {
                    "asset": "USDC",
                    "method": "Arbitrum One Network - Native",
                    "address": _WALLET,
                    "key": "unverified_hl",
                    "verified": False,
                }
            ],
            cfg,
        )

    # Rejects wrong destination wallet
    with pytest.raises(ValueError, match="expected Hyperliquid wallet"):
        reb.resolve_verified_withdraw_address(
            [
                {
                    "asset": "USDC",
                    "method": "Arbitrum One Network - Native",
                    "address": _OTHER_WALLET,
                    "key": "other_wallet",
                    "verified": True,
                }
            ],
            cfg,
        )

    # Accepts exact verified Native Arbitrum One USDC address (case-insensitive hex match)
    chosen = reb.resolve_verified_withdraw_address(
        [
            {
                "asset": "USDC",
                "method": "Arbitrum One Network",
                "address": _WALLET,
                "key": "bridged_ignored",
                "verified": True,
            },
            {
                "asset": "USDC",
                "method": "Arbitrum One Network - Native",
                "address": _WALLET.lower(),
                "key": "HL_ARB_MAIN",
                "verified": True,
            },
        ],
        cfg,
    )
    assert chosen["key"] == "HL_ARB_MAIN"


def _mock_client(*, zusd: float = 6400.0, usdc: float = 20.0, fee: float = 1.0) -> MagicMock:
    cli = MagicMock(spec=KrakenClient)
    cli.balance.return_value = {"ZUSD": str(zusd), "USDC": str(usdc)}
    cli.withdraw_methods.return_value = [
        {
            "asset": "USDC",
            "method": "Arbitrum One Network - Native",
            "network": "Arbitrum One (USDC)",
            "minimum": "4.00",
            "fee": {"asset": "USDC", "fee": str(fee)},
        }
    ]
    cli.withdraw_addresses.return_value = [
        {
            "asset": "USDC",
            "method": "Arbitrum One Network - Native",
            "address": _WALLET,
            "key": "HL_ARB_MAIN",
            "verified": True,
        }
    ]

    def _winfo(asset, key, amount):
        return {
            "method": "Arbitrum One Network - Native",
            "limit": "500000.0",
            "amount": f"{amount - fee:.4f}",
            "fee": f"{fee:.4f}",
        }

    cli.withdraw_info.side_effect = _winfo
    cli.withdraw.return_value = {"refid": "REF-TEST-123"}
    cli.withdraw_status.return_value = [
        {"refid": "REF-TEST-123", "status": "Success", "txid": "0xarbtx123"}
    ]
    return cli


def test_plan_and_run_dry_run_and_min_net_guard():
    cfg = _make_cfg()
    cli = _mock_client(zusd=6400.0, usdc=20.0, fee=1.0)

    with (
        patch.object(reb, "fetch_arbitrum_balances", return_value={"eth": 0.0012, "usdc": 0.0}),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
    ):
        # Dry-run never invokes withdraw or add_order
        res = reb.plan_and_run_transfer(cli, cfg, amount_usdc=11.0, execute=False)
        assert res["mode"] == "dry_run"
        assert res["plan"]["net_arbitrum_usdc"] == pytest.approx(10.0)
        assert res["plan"]["withdraw_key"] == "HL_ARB_MAIN"
        cli.withdraw.assert_not_called()
        cli.add_order.assert_not_called()

        # Net receipt below 6.0 USDC (Bridge2 < 5 USDC loss trap) is blocked
        with pytest.raises(ValueError, match="KRAKEN_HL_MIN_NET_USDC"):
            reb.plan_and_run_transfer(cli, cfg, amount_usdc=5.5, execute=False)


def test_execute_refuses_shortfall_without_convert_flag_and_executes_when_funded():
    cfg = _make_cfg()
    cli_short = _mock_client(zusd=6400.0, usdc=1.5, fee=1.0)

    with (
        patch.object(reb, "fetch_arbitrum_balances", return_value={"eth": 0.0012, "usdc": 0.0}),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
    ):
        with pytest.raises(ValueError, match="Kraken UI Convert"):
            reb.plan_and_run_transfer(cli_short, cfg, amount_usdc=11.0, execute=True, convert_usd=False)
        cli_short.withdraw.assert_not_called()

    cli_funded = _mock_client(zusd=6400.0, usdc=25.0, fee=1.0)
    arb_seq = [{"eth": 0.0012, "usdc": 0.0}, {"eth": 0.0012, "usdc": 10.0}]
    with (
        patch.object(reb, "fetch_arbitrum_balances", side_effect=arb_seq),
        patch.object(
            reb,
            "fetch_hyperliquid_balances",
            return_value={"spot_usdc": 54.0, "spot_hype": 15.9, "perp_equity_usd": 0.0, "perp_withdrawable_usd": 0.0},
        ),
    ):
        res = reb.plan_and_run_transfer(cli_funded, cfg, amount_usdc=11.0, execute=True, wait_arb=True)
        assert res["mode"] == "executed"
        assert res["refid"] == "REF-TEST-123"
        assert res["arbitrum_arrival"]["arrived"] is True
        assert res["arbitrum_arrival"]["arb_usdc_delta"] == pytest.approx(10.0)
        cli_funded.withdraw.assert_called_once_with(
            "USDC", "HL_ARB_MAIN", 11.0, address=_WALLET, max_fee=1.25
        )
