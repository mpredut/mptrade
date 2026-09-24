import os

from botcore import load_env_stack


def test_env_stack_preserves_process_secret_and_versioned_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / "profile.env"
    config_file = tmp_path / "config.env"
    env_file.write_text(
        "STACK_SHARED=secret-file\nSTACK_SECRET_ONLY=secret\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "STACK_SHARED=versioned\nSTACK_CONFIG_ONLY=config\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STACK_SHARED", "process")
    monkeypatch.delenv("STACK_SECRET_ONLY", raising=False)
    monkeypatch.delenv("STACK_CONFIG_ONLY", raising=False)

    load_env_stack(str(env_file))

    assert os.environ["STACK_SHARED"] == "process"
    assert os.environ["STACK_SECRET_ONLY"] == "secret"
    assert os.environ["STACK_CONFIG_ONLY"] == "config"


def test_env_stack_resolves_config_next_to_selected_env_file(tmp_path, monkeypatch):
    nested = tmp_path / "venue"
    nested.mkdir()
    env_file = nested / ".env"
    env_file.write_text("", encoding="utf-8")
    (nested / "runtime.env").write_text("STACK_ADJACENT=yes\n", encoding="utf-8")
    monkeypatch.delenv("STACK_ADJACENT", raising=False)

    load_env_stack(str(env_file), "runtime.env")

    assert os.environ["STACK_ADJACENT"] == "yes"


def test_venue_env_stack_overrides_root_config_env(tmp_path, monkeypatch):
    import botcore

    # Simulate root config.env loading a fallback key
    root_cfg = tmp_path / "root_config.env"
    root_cfg.write_text("VENUE_FALLBACK=root_val\nVENUE_ROOT_ONLY=root_only\n", encoding="utf-8")
    monkeypatch.setattr(botcore, "_ROOT_CONFIG_PATH", str(root_cfg.resolve()))
    monkeypatch.delenv("VENUE_FALLBACK", raising=False)
    monkeypatch.delenv("VENUE_ROOT_ONLY", raising=False)
    monkeypatch.delenv("VENUE_CFG_ONLY", raising=False)

    botcore.load_dotenv(str(root_cfg))
    assert os.environ["VENUE_FALLBACK"] == "root_val"
    assert os.environ["VENUE_ROOT_ONLY"] == "root_only"

    # Venue stack with overriding value
    venue_dir = tmp_path / "venue"
    venue_dir.mkdir()
    venue_cfg = venue_dir / "config.env"
    venue_cfg.write_text("VENUE_FALLBACK=venue_val\nVENUE_CFG_ONLY=venue_only\n", encoding="utf-8")
    venue_env = venue_dir / ".env"
    venue_env.write_text("", encoding="utf-8")

    load_env_stack(str(venue_env))

    # Venue config takes precedence over root config.env
    assert os.environ["VENUE_FALLBACK"] == "venue_val"
    assert os.environ["VENUE_CFG_ONLY"] == "venue_only"
    # Root keys not overridden by venue remain available as defaults
    assert os.environ["VENUE_ROOT_ONLY"] == "root_only"

