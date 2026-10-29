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


# ---------------------------------------------------------------------------
# Cross-Venue Tradable Instrument Discovery, Resolution & Live Price Lookup
# ---------------------------------------------------------------------------

def discover_kraken_cache_pairs(workspace_dir: str) -> list[tuple[str, str, str]]:
    """Read KRAKEN_TRADABLE_PAIRS (or fallback KRAKEN_CACHE_PAIRS) and listed xStock pairs from kraken/."""
    import os
    from botcore import parse_env_file
    from cachepaths import split_symbol_assets

    kv = parse_env_file(os.path.join(workspace_dir, "kraken", "config.env"))
    raw_pairs = kv.get("KRAKEN_TRADABLE_PAIRS") or kv.get("KRAKEN_CACHE_PAIRS") or ""
    out: list[tuple[str, str, str]] = []
    seen_pairs: set[str] = set()
    for item in raw_pairs.split(","):
        pair = item.strip().upper()
        if not pair or pair in seen_pairs:
            continue
        split = split_symbol_assets(pair)
        if split is not None:
            seen_pairs.add(pair)
            out.append((pair, split[0], split[1]))
    return out


def discover_hyperliquid_spot(workspace_dir: str) -> list[tuple[str, str, str]]:
    """Read hyperliquid/config.env to include live Hyperliquid spot pairs (e.g. HYPEUSDC)."""
    import os
    from botcore import _parse_bool_flag, load_env_file_defaults

    cfg_path = os.path.join(workspace_dir, "hyperliquid", "config.env")
    if not os.path.exists(cfg_path):
        return []
    kv = load_env_file_defaults(cfg_path)
    if not _parse_bool_flag(kv.get("HL_LIVE_ORDERS"), False):
        return []
    token = (kv.get("HL_SPOT_TOKEN") or "HYPE").strip().upper()
    if not token:
        return []
    return [(f"{token}USDC", token, "USDC")]


def discover_t212_equities(workspace_dir: str) -> list[tuple[str, str, str, str]]:
    """Read 212trading/config.*.env to discover configured Trading212 equity tickers."""
    import glob
    import os
    from botcore import parse_env_file

    t212_dir = os.path.join(workspace_dir, "212trading")
    if not os.path.isdir(t212_dir):
        return []
    out: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for cfg_path in sorted(glob.glob(os.path.join(glob.escape(t212_dir), "config.*.env"))):
        kv = parse_env_file(cfg_path)
        ticker = (kv.get("T212_TICKER") or kv.get("STRAT_PAIR") or "").upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            short = ticker.split("_", 1)[0]
            out.append((f"T212_{short}", ticker, ticker, "USD"))
    return out


def load_tradable_instruments(
    workspace_dir: str | None = None,
    *,
    only_workspace_conf: bool = False,
    include_closed_rth: bool = False,
    now_ts: float | None = None,
    asset_scope: str = "all",
    is_rth_open_fn=None,
) -> dict:
    """Load enabled instruments across Binance, Kraken, Hyperliquid, and Trading212 filtered by asset_scope."""
    import os
    from botcore import is_us_rth_open, load_env_file_defaults

    ws_dir = workspace_dir or os.path.dirname(os.path.abspath(__file__))
    rth_check = is_rth_open_fn or is_us_rth_open
    scope = (asset_scope or "all").strip().lower()
    ws_inst_conf = os.path.join(ws_dir, "instruments.conf")
    raw_map: dict = {}
    if os.path.exists(ws_inst_conf) or not only_workspace_conf:
        try:
            raw_map = load_instruments(ws_inst_conf) if os.path.exists(ws_inst_conf) else load_instruments()
        except Exception:
            return {}

    active_map: dict = {}

    def _existing_syms(prov: str) -> set[str]:
        return {
            str(getattr(inst, "symbol", "")).upper()
            for inst in active_map.values()
            if str(getattr(inst, "provider_name", "")).lower() == prov
        }

    def _bind_inst(
        name: str, prov: str, sym: str, base: str, quote: str,
        *, live: bool = False, validity: str | None = None, **kw,
    ) -> bool:
        try:
            from instrument import Instrument
            inst = Instrument(name=name, provider=prov, symbol=sym, base=base, quote=quote, enabled=True, **kw)
            if hasattr(inst, "_provider"):
                if live:
                    inst._provider._live_enabled = True
                if validity:
                    inst._provider._order_validity = validity
            active_map[name] = inst
            return True
        except Exception:
            return False

    if scope in ("all", "crypto"):
        for k, inst in raw_map.items():
            if getattr(inst, "enabled", True) is False:
                continue
            if scope == "crypto" and str(getattr(inst, "provider_name", "")).lower() == "t212":
                continue
            active_map[k] = inst

        existing_kraken_syms = _existing_syms("kraken")
        for pair, base, quote in discover_kraken_cache_pairs(ws_dir):
            if pair not in existing_kraken_syms and _bind_inst(f"KRAKEN_{base}", "kraken", pair, base, quote):
                existing_kraken_syms.add(pair)

        existing_hl_syms = _existing_syms("hyperliquid")
        for pair, base, quote in discover_hyperliquid_spot(ws_dir):
            if (
                pair not in existing_hl_syms
                and base not in existing_hl_syms
                and _bind_inst(f"HYPERLIQUID_{base}", "hyperliquid", pair, base, quote, live=True)
            ):
                existing_hl_syms.add(pair)

    if scope in ("all", "equities") and (include_closed_rth or rth_check(now_ts)):
        load_env_file_defaults(os.path.join(ws_dir, "212trading", "runtime.env"))
        existing_t212_syms = _existing_syms("t212")
        for inst_name, ticker, base, quote in discover_t212_equities(ws_dir):
            if ticker not in existing_t212_syms and _bind_inst(
                inst_name, "t212", ticker, base, quote, live=True, validity="DAY", market_hours="rth"
            ):
                existing_t212_syms.add(ticker)

    return active_map


def resolve_tradable_instrument(
    symbol: str,
    venue: str | None = None,
    instruments_map: dict | None = None,
    *,
    workspace_dir: str | None = None,
):
    """Resolve an enabled Instrument instance for the given symbol and venue."""
    from cachepaths import canonical_base_asset, normalize_venue_name
    from intelligence.macro_sentiment_ssot import clean_symbol_key

    if instruments_map is None:
        instruments_map = load_tradable_instruments(workspace_dir)

    sym_norm = clean_symbol_key(symbol)
    if not sym_norm or isinstance(venue, bool):
        return None
    venue_norm = normalize_venue_name(venue) or None
    req_base = canonical_base_asset(symbol)

    candidates = [
        inst for inst in instruments_map.values()
        if getattr(inst, "enabled", True) is not False
        and (not venue_norm or normalize_venue_name(str(getattr(inst, "provider_name", ""))) == venue_norm)
    ]

    for inst in candidates:
        if clean_symbol_key(inst.symbol) == sym_norm:
            return inst
    if req_base:
        for inst in candidates:
            raw_b = getattr(inst, "base", None) or getattr(inst, "symbol", "")
            if canonical_base_asset(str(raw_b)) == req_base:
                return inst
    for inst in candidates:
        inst_sym_n = clean_symbol_key(inst.symbol)
        if inst_sym_n.startswith(sym_norm) or sym_norm.startswith(inst_sym_n):
            return inst
    return None


def resolve_instrument_live_price(inst, cache_dir: str = "cachedb") -> float:
    """Fetch live market price for ``inst`` from provider or ``cachedb`` fallbacks."""
    import json
    import os
    from botcore import extract_current_price_item, safe_pos_float
    from cachepaths import cache_symbol_variants, canonical_base_asset, split_symbol_assets

    try:
        if hasattr(inst._provider, "get_current_price"):
            px_f = safe_pos_float(inst._provider.get_current_price(inst.symbol), default=None)
            if px_f is not None:
                return px_f
    except Exception:
        pass

    cache_path = os.path.join(cache_dir, "cache_current_price.json")
    try:
        from state_io import read_recovering_previous
        cdata = read_recovering_previous(cache_path, json.loads, default=None)
        if isinstance(cdata, dict):
            raw_items = cdata.get("items")
            items = raw_items if isinstance(raw_items, dict) else {}
            sym_u = str(inst.symbol or "").strip().upper()
            candidates = cache_symbol_variants(inst.symbol, include_original=True)
            inst_base = getattr(inst, "base", None) or (split_symbol_assets(sym_u) or (sym_u, ""))[0]
            if inst_base:
                for suffix in ("USDC", "USDT", "USD"):
                    cand = f"{str(inst_base).upper()}{suffix}"
                    if cand not in candidates:
                        candidates.append(cand)

            for cand in candidates:
                px_f = extract_current_price_item(items.get(cand))
                if px_f is not None:
                    return px_f
            if inst_base:
                base_u = canonical_base_asset(str(inst_base))
                for k, v in items.items():
                    if canonical_base_asset(str(k)) == base_u and (px_f := extract_current_price_item(v)) is not None:
                        return px_f
    except Exception:
        pass

    try:
        from intelligence.sentiment.guards.high_stake_guard import load_symbol_price_series
        ser_24h, _ = load_symbol_price_series(cache_dir, str(inst.symbol or ""), include_7d=False)
        if ser_24h:
            px_f = safe_pos_float(ser_24h[-1][1], default=None)
            if px_f is not None:
                return px_f
    except Exception:
        pass

    return float("nan")
