"""Shared-memory utilities built on ``multiprocessing.shared_memory``.

shmutils — read/write connection to a named segment plus JSON serialization in
           a fixed buffer. Usage:
               from shmem import shmutils

The package name is distinct from the base ``utils.py`` module to avoid shadowing it.
"""
