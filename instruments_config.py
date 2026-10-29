"""Bind validated, credential-free registry records to the provider facade."""
from dataclasses import asdict

from instrument_registry import load_registry, select_instruments


def _bind(specs, api):
    # Import only at the binding boundary: symbols/cache discovery needs no API.
    from instrument import Instrument
    return {name: Instrument(**asdict(spec), api=api) for name, spec in specs.items()}


def load_instruments(path=None, api=None):
    return _bind(load_registry(path), api)


def load_for(consumer, path=None, api=None, only_enabled=True):
    """Bind instruments explicitly enabled for the requested consumer role."""
    return _bind(select_instruments(role=consumer, path=path,
                                    only_enabled=only_enabled), api)


# Compatibility APIs delegate to the same validated registry, never a second parser.
def binance_symbols(path=None):
    from instrument_registry import symbols_for
    symbols = symbols_for("binance", path=path)
    if not symbols:
        raise ValueError("instruments.conf: no enabled Binance instruments")
    return symbols


def trail_pct_map(path=None):
    return {spec.symbol: spec.number("trailing.pct") for spec in
            select_instruments("binance", "trailing", path=path).values()}


def tradeall_trade_symbols(path=None):
    from instrument_registry import symbols_for
    return set(symbols_for("binance", "tradeall_fire", path=path))
