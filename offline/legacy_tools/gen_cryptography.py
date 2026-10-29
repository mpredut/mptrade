#!/usr/bin/env python3
"""Generate an Ed25519 key pair. Historical helper, run manually.

This file used to be a shell one-liner (``python -c "..."``) saved with a .py
extension, so it did not parse as Python at all: any lint, import or AST pass over
offline/legacy_tools crashed on it. The body was already valid Python, so the
wrapper is gone and the script runs directly:

    python3 offline/legacy_tools/gen_cryptography.py

It writes ed25519_private.pem and ed25519_public.pem into the current directory and
prints the public key in base64.
"""
import base64
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from state_io import atomic_write_bytes

def generate_ed25519_keypair(priv_path: str = 'ed25519_private.pem', pub_path: str = 'ed25519_public.pem') -> str:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    pub_bytes = public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    priv_bytes = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())

    atomic_write_bytes(priv_path, priv_bytes)
    atomic_write_bytes(pub_path, pub_bytes)

    return base64.b64encode(pub_bytes).decode()


if __name__ == "__main__":
    pub_b64 = generate_ed25519_keypair()
    print('Keys generated: ed25519_private.pem and ed25519_public.pem')
    print('Public key base64:')
    print(pub_b64)
