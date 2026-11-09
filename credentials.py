"""Explicit credential-pair selection without API or client construction.

The resolver never combines a key from one profile with a secret from another.
A partially configured higher-priority profile is an error rather than a reason
to hide the mistake by falling back to another account.
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from botcore import load_dotenv

_existing_keys = sys.modules.get("keys")
_keys_blocked = (
    _existing_keys is not None
    and not hasattr(_existing_keys, "__path__")
    and not getattr(_existing_keys, "_is_credentials_shim", False)
) or ("keys" in sys.modules and _existing_keys is None)
if not _keys_blocked:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

api_key = os.environ.get("BINANCE_API_KEY", "").strip()
api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
# The Ed25519 stream key is distinct from the HMAC REST credential.
api_key_ws = os.environ.get("BINANCE_API_KEY_WS", "").strip()

if not _keys_blocked:
    if _existing_keys is None:
        _existing_keys = types.ModuleType("keys")
        _existing_keys.__path__ = []
        _existing_keys._is_credentials_shim = True
        sys.modules["keys"] = _existing_keys
    if getattr(_existing_keys, "__spec__", None) is None:
        try:
            from botcore import _MergedModuleAliasFinder
            _existing_keys.__spec__ = _MergedModuleAliasFinder.find_spec("keys")
        except Exception:
            pass
    _existing_keys.apikeys = sys.modules[__name__]
    sys.modules["keys.apikeys"] = sys.modules[__name__]


if not isinstance(globals().get("CredentialConfigurationError"), type) or not issubclass(
    globals()["CredentialConfigurationError"], RuntimeError
):
    class CredentialConfigurationError(RuntimeError):
        """A credential profile is missing, incomplete, or ambiguous."""


if not isinstance(globals().get("CredentialProfileMissingError"), type) or not issubclass(
    globals()["CredentialProfileMissingError"], CredentialConfigurationError
):
    class CredentialProfileMissingError(CredentialConfigurationError):
        """None of the approved profiles is configured."""


@dataclass(frozen=True)
class CredentialPair:
    key: str
    secret: str
    profile: str


def _clean_cred_val(raw: object) -> str:
    return "" if (raw is None or isinstance(raw, bool)) else str(raw).strip()


def resolve_credential_pair(
    profiles: Sequence[tuple[str, str, str]], *,
    values: Mapping[str, str] | None = None,
) -> CredentialPair:
    """Return the first complete profile, rejecting every partial profile."""
    if values is not None and not isinstance(values, Mapping):
        raise CredentialConfigurationError("Credential values must be a mapping")
    if isinstance(profiles, (str, bytes)) or not isinstance(profiles, Sequence) or not profiles:
        raise CredentialConfigurationError("Credential profiles must be a non-empty sequence")
    source = os.environ if values is None else values
    labels_seen: list[str] = []
    for item in profiles:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 3:
            raise CredentialConfigurationError("Each credential profile must be a (label, key_name, secret_name) tuple")
        label_str, key_var, secret_var = (str(x or "").strip() for x in item)
        if not label_str or not key_var or not secret_var:
            raise CredentialConfigurationError("Credential profile label, key_name, and secret_name must be non-empty")
        labels_seen.append(label_str)
        key = _clean_cred_val(source.get(key_var, ""))
        secret = _clean_cred_val(source.get(secret_var, ""))
        if bool(key) != bool(secret):
            raise CredentialConfigurationError(
                f"Incomplete {label_str} credentials: missing {secret_var if key else key_var}"
            )
        if key:
            return CredentialPair(key=key, secret=secret, profile=label_str)
    raise CredentialProfileMissingError(f"No complete credential profile: {', '.join(labels_seen)}")


_KRAKEN_BOT_FALLBACK = ("Kraken BOT fallback", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT")
_KRAKEN_SPARE = ("Kraken SPARE", "KRAKEN_API_KEY_SPARE", "KRAKEN_API_SECRET_SPARE")

_KRAKEN_PROFILES = {
    "bot": (("Kraken BOT", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"),),
    "cache": (("Kraken CACHE", "KRAKEN_API_KEY_CACHE", "KRAKEN_API_SECRET_CACHE"), _KRAKEN_BOT_FALLBACK),
    "trail": (("Kraken TRAIL", "KRAKEN_API_KEY_TRAIL", "KRAKEN_API_SECRET_TRAIL"), _KRAKEN_BOT_FALLBACK),
    "fleet": (("Kraken primary", "KRAKEN_API_KEY", "KRAKEN_API_SECRET"), _KRAKEN_SPARE, _KRAKEN_BOT_FALLBACK),
    "analysis": (_KRAKEN_SPARE, _KRAKEN_BOT_FALLBACK),
}


def kraken_credentials(role: str, *, values=None) -> CredentialPair:
    """Resolve an approved Kraken role and its explicitly ordered fallbacks."""
    role_key = "" if isinstance(role, bool) else str(role or "").strip().lower()
    if role_key not in _KRAKEN_PROFILES:
        raise ValueError(f"Unknown Kraken credential role: {role}")
    return resolve_credential_pair(_KRAKEN_PROFILES[role_key], values=values)


def t212_credentials(*, values=None) -> CredentialPair:
    """Resolve the single T212 API key/secret pair."""
    return resolve_credential_pair((("T212", "T212_API_KEY", "T212_API_SECRET"),), values=values)
