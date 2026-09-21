"""Facade for backward compatibility with alertnotifiers."""
import sys
from notify_engine import alertnotifiers
sys.modules[__name__] = alertnotifiers
