"""Binance API credentials, sourced from .env (the single secret store).

This module holds no secrets. It re-exports BINANCE_API_KEY / BINANCE_API_SECRET
/ BINANCE_API_KEY_WS from the environment, loading the repo .env defensively so
importers that have not already loaded it still receive the keys. Read-only imports
need no credentials. Authenticated clients must validate them before construction.
"""
import os

from botcore import load_dotenv

load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

api_key = os.environ.get("BINANCE_API_KEY", "").strip()
api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
# The Ed25519 stream key is distinct from the HMAC REST credential.
api_key_ws = os.environ.get("BINANCE_API_KEY_WS", "").strip()
