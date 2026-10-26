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
    source = os.environ if values is None else values
    for label, key_name, secret_name in profiles:
        raw_k = source.get(key_name, "")
        raw_s = source.get(secret_name, "")
        key = "" if (raw_k is None or isinstance(raw_k, bool)) else str(raw_k).strip()
        secret = "" if (raw_s is None or isinstance(raw_s, bool)) else str(raw_s).strip()
        if bool(key) != bool(secret):
            missing = secret_name if key else key_name
            raise CredentialConfigurationError(
                f"Incomplete {label} credentials: missing {missing}"
            )
        if key:
            return CredentialPair(key=key, secret=secret, profile=label)
    labels = ", ".join(label for label, _key, _secret in profiles)
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
    try:
        profiles = _KRAKEN_PROFILES[role]
    except KeyError as exc:
        raise ValueError(f"Unknown Kraken credential role: {role}") from exc
    return resolve_credential_pair(profiles, values=values)


def t212_credentials(*, values=None) -> CredentialPair:
    """Resolve the single T212 API key/secret pair."""
    return resolve_credential_pair((
        ("T212", "T212_API_KEY", "T212_API_SECRET"),
    ), values=values)
