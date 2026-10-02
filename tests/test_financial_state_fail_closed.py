import os

import pytest

from assetguardian_state import AssetGuardianState
from state_io import StateReadError
from trailing_core import TrailingCore


def test_assetguardian_initializes_its_state_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for load_first in (False, True):
        for relative in (False, True):
            sub = tmp_path / f"dir_{load_first}_{relative}"
            sub.mkdir(parents=True, exist_ok=True)
            monkeypatch.chdir(sub)
            path = "assetguardian.json" if relative else str(sub / "nested" / "state.json")
            state = AssetGuardianState(path)
            if load_first:
                assert state.load() == {}
            value = {"version": 2, "symbols": {"TEST": {"pending": True}}}
            state.save(value)
            assert state.load() == value


def test_assetguardian_rejects_corrupt_live_state(tmp_path):
    path = tmp_path / "assetguardian.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(StateReadError, match="AssetGuardian"):
        AssetGuardianState(str(path)).load()


def test_assetguardian_reset_requires_explicit_non_live_policy(tmp_path):
    path = tmp_path / "assetguardian.json"
    path.write_text("[]", encoding="utf-8")
    assert AssetGuardianState(str(path), fail_closed=False).load() == {}


def test_live_trailing_rejects_corrupt_state_but_paper_can_reset(tmp_path):
    path = tmp_path / "trailing.json"
    path.write_text("{broken", encoding="utf-8")
    common = dict(
        adapter=object(), log=lambda _message: None, state_file=str(path),
        min_notional=1.0, rebuy_enabled=False, rebuy_bounce_pct=1.0,
        rebuy_skip_if_trend_down=False, sell_skip_if_trend_up=False,
    )
    live = TrailingCore(enabled=True, **common)
    with pytest.raises(StateReadError, match="live trailing"):
        live.load()
    assert TrailingCore(enabled=False, **common).load() == {}


def test_hyperliquid_live_engines_declare_fail_closed_reads():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for relative in ("hyperliquid/strategy.py", "hyperliquid/archive/delta_neutral/delta_neutral.py"):
        text = open(os.path.join(root, relative), encoding="utf-8").read()
        assert "fail_closed=not self.dry_run" in text
