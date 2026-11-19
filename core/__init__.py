"""Canonical Layer-0 core primitives package.

Submodules:
- ``core.lock``: logging and cross-process ``FileLock`` / ``single_instance``
- ``core.math_coercion``: scalar coercion, order extraction, and deterministic math
- ``core.config``: ``Config`` / ``cfg``, dotenv/INI parsing, typed env & CLI parsers
- ``core.http``: stdlib & ``requests`` HTTP gateways and Ed25519 signing
- ``core.time_market``: epoch normalization, US RTH hours, and price-freshness SLA
- ``core.heartbeat_runtime``: heartbeats, ``procs.conf`` manifest, ``RuntimeServices``, and loaders
- ``core.llm_topics``: declarative ``TopicSpec`` and ``TOPIC_SPECS`` registry
"""
from __future__ import annotations
