"""Compatibility wrapper for tradeall_observe."""
import sys
from visuals import tradeall_observe
sys.modules[__name__] = tradeall_observe

if __name__ == "__main__":
    tradeall_observe.main()

