import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
T212_DIR = ROOT / "212trading"
if str(T212_DIR) not in sys.path:
    sys.path.insert(0, str(T212_DIR))

ipo_common = importlib.import_module("ipo_common")


def test_t212_environment_uses_one_documented_layer_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ipo_common, "load_dotenv", calls.append)
    env_file = tmp_path / ".env"
    profile_file = tmp_path / "config.asset.env"

    ipo_common.load_t212_environment(
        str(env_file), profile_file=str(profile_file),
    )

    assert calls == [
        os.path.abspath(env_file),
        str(ROOT / ".env"),
        os.path.abspath(profile_file),
        str(T212_DIR / "runtime.env"),
    ]


def test_t212_launchers_use_the_shared_environment_loader():
    for relative in ("ipo.py", "t212_bot.py", "t212_status.py", "price_alert.py"):
        text = (T212_DIR / relative).read_text(encoding="utf-8")
        assert "load_t212_environment(" in text


def test_t212_environment_precedence_is_effective(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    t212 = root / "212trading"
    t212.mkdir(parents=True)
    (root / ".env").write_text("LAYER=shared\nSHARED_ONLY=yes\n", encoding="utf-8")
    secrets = t212 / ".env"
    secrets.write_text("LAYER=t212\nT212_ONLY=yes\n", encoding="utf-8")
    profile = t212 / "config.asset.env"
    profile.write_text("LAYER=profile\nPROFILE_ONLY=yes\n", encoding="utf-8")
    (t212 / "runtime.env").write_text(
        "LAYER=runtime\nRUNTIME_ONLY=yes\n", encoding="utf-8",
    )
    monkeypatch.setattr(ipo_common, "__file__", str(t212 / "ipo_common.py"))

    with monkeypatch.context() as context:
        context.setattr(ipo_common, "__file__", str(t212 / "ipo_common.py"))
        context.setenv("LAYER", "process")
        for key in ("SHARED_ONLY", "T212_ONLY", "PROFILE_ONLY", "RUNTIME_ONLY"):
            context.delenv(key, raising=False)
        ipo_common.load_t212_environment(str(secrets), profile_file=str(profile))
        assert os.environ["LAYER"] == "process"
        assert os.environ["T212_ONLY"] == "yes"
        assert os.environ["SHARED_ONLY"] == "yes"
        assert os.environ["PROFILE_ONLY"] == "yes"
        assert os.environ["RUNTIME_ONLY"] == "yes"
