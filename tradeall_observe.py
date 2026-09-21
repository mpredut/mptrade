"""Compatibility wrapper for tradeall_observe."""
import sys
from visuals import tradeall_observe
sys.modules[__name__] = tradeall_observe
