"""Explicit credential-pair selection without API or client construction.

The resolver never combines a key from one profile with a secret from another.
A partially configured higher-priority profile is an error rather than a reason
to hide the mistake by falling back to another account.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


class CredentialConfigurationError(RuntimeError):
    """A credential profile is missing, incomplete, or ambiguous."""


class CredentialProfileMissingError(CredentialConfigurationError):
    """None of the approved profiles is configured."""


@dataclass(frozen=True)
class CredentialPair:
    key: str
    secret: str
    profile: str


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
        label, key_name, secret_name = item
        label_str = str(label or "").strip()
        key_var = str(key_name or "").strip()
        secret_var = str(secret_name or "").strip()
        if not label_str or not key_var or not secret_var:
            raise CredentialConfigurationError("Credential profile label, key_name, and secret_name must be non-empty")
        labels_seen.append(label_str)
        raw_k = source.get(key_var, "")
        raw_s = source.get(secret_var, "")
        key = "" if (raw_k is None or isinstance(raw_k, bool)) else str(raw_k).strip()
        secret = "" if (raw_s is None or isinstance(raw_s, bool)) else str(raw_s).strip()
        if bool(key) != bool(secret):
            missing = secret_var if key else key_var
            raise CredentialConfigurationError(
                f"Incomplete {label_str} credentials: missing {missing}"
            )
        if key:
            return CredentialPair(key=key, secret=secret, profile=label_str)
    labels = ", ".join(labels_seen)
    raise CredentialProfileMissingError(f"No complete credential profile: {labels}")


_KRAKEN_PROFILES = {
    "bot": (
        ("Kraken BOT", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"),
    ),
    "cache": (
        ("Kraken CACHE", "KRAKEN_API_KEY_CACHE", "KRAKEN_API_SECRET_CACHE"),
        ("Kraken BOT fallback", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"),
    ),
    "trail": (
        ("Kraken TRAIL", "KRAKEN_API_KEY_TRAIL", "KRAKEN_API_SECRET_TRAIL"),
        ("Kraken BOT fallback", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"),
    ),
    "fleet": (
        ("Kraken primary", "KRAKEN_API_KEY", "KRAKEN_API_SECRET"),
        ("Kraken SPARE", "KRAKEN_API_KEY_SPARE", "KRAKEN_API_SECRET_SPARE"),
        ("Kraken BOT fallback", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"),
    ),
    "analysis": (
        ("Kraken SPARE", "KRAKEN_API_KEY_SPARE", "KRAKEN_API_SECRET_SPARE"),
        ("Kraken BOT fallback", "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT"),
    ),
}


def kraken_credentials(role: str, *, values=None) -> CredentialPair:
    """Resolve an approved Kraken role and its explicitly ordered fallbacks."""
    role_key = str(role or "").strip().lower() if not isinstance(role, bool) else ""
    try:
        profiles = _KRAKEN_PROFILES[role_key]
    except KeyError as exc:
        raise ValueError(f"Unknown Kraken credential role: {role}") from exc
    return resolve_credential_pair(profiles, values=values)


def t212_credentials(*, values=None) -> CredentialPair:
    """Resolve the single T212 API key/secret pair."""
    return resolve_credential_pair((
        ("T212", "T212_API_KEY", "T212_API_SECRET"),
    ), values=values)
