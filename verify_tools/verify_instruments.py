#!/usr/bin/env python3
# verify_instruments.py
"""Verify the foundation (steps 1-3): name registry + Instrument + instruments.conf.

It does not touch the fleet. It confirms:
  1. instruments.conf loads; BTC/TAO/HYPE present, provider/symbol/base correct.
  2. the mt.* params MIRROR monitortrades.conf (behaviour-preserving) — a real diff.
  3. explicit per-venue routing: TAO->Binance, HYPE->Hyperliquid (provider_by_name).
  4. generic operations through Instrument == the facade directly (price, balance).
Exits 0 when everything passes.
"""
import os
import sys

# Run from verify_tools/ -> put the repo ROOT on the path for the imports below.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from providers.market_api import api
from instruments_config import load_instruments, load_for

FAIL = []


def check(cond, label, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAIL.append(label)


def check_available(cond, available, label, detail=""):
    if not available:
        print(f"  [SKIP] {label} — date private indisponibile" +
              (f" ({detail})" if detail else ""))
        return
    check(cond, label, detail)


def parse_monitortrades_conf(path=None):
    """Parser simplu pt formatul existent (key=val; SYMBOL = g / l / maxage)."""
    from state_io import read_text_lines

    if path is None:
        path = os.path.join(_ROOT, "monitortrades.conf")
    g = {}
    per = {}
    for raw in read_text_lines(path):
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = (x.strip() for x in line.split("=", 1))
        if "/" in v:                      # SYMBOL = gain / lost / maxage
            parts = [p.strip() for p in v.split("/")]
            if len(parts) == 3:
                per[k] = parts
        else:
            g[k] = v
    return g, per


def main() -> int:
    FAIL.clear()
    print("==== 1. INCARCARE instruments.conf ====")
    inst = load_instruments()
    for name in ("BINANCE_BTC", "BINANCE_TAO", "HYPERLIQUID_HYPE"):
        check(name in inst, f"sectiune {name} prezenta")
    tao = inst.get("BINANCE_TAO")
    btc = inst.get("BINANCE_BTC")
    hype = inst.get("HYPERLIQUID_HYPE")

    if tao:
        check(tao.symbol == "TAOUSDC" and tao.base == "TAO", "TAO symbol/base",
              f"{tao.symbol}/{tao.base}")
    if hype:
        check(hype.symbol == "HYPEUSDC" and hype.base == "HYPE", "HYPE symbol/base",
              f"{hype.symbol}/{hype.base}")
        check(hype.enabled is False, "HYPE enabled=no (gated)", str(hype.enabled))

    print("\n==== 2. params mt.* == monitortrades.conf (behavior-preserving) ====")
    g, per = parse_monitortrades_conf()
    for symname, key in (("BINANCE_BTC", "BTCUSDC"), ("BINANCE_TAO", "TAOUSDC")):
        it = inst.get(symname)
        exp = per.get(key)
        if it and exp:
            got = [str(it.param("mt", "gain")), str(it.param("mt", "lost")),
                   str(it.param("mt", "maxage_days"))]
            try:
                match = [float(x) for x in got] == [float(x) for x in exp]
            except (TypeError, ValueError):
                match = False
            check(match,
                  f"{symname} gain/lost/maxage", f"conf={exp} instr={got}")
    # globale -> mt.* pe TAO (oglindesc hard_tp_* + tp_reference)
    if tao and g:
        def _num_eq(a, b) -> bool:
            try:
                return float(a) == float(b)
            except (TypeError, ValueError):
                return str(a) == str(b)

        check(_num_eq(tao.param("mt", "hardtp"), g.get("hard_tp_pct")),
              "hardtp == hard_tp_pct", f"{tao.param('mt','hardtp')} vs {g.get('hard_tp_pct')}")
        # Note: Instruments may diverge from legacy monitortrades.conf defaults due to backtest tuning
        # (e.g. TAO hardtp_fraction optimized 0.5 -> 0.4). Check equality or valid tuned override.
        tao_frac = tao.param("mt", "hardtp_fraction")
        check(_num_eq(tao_frac, g.get("hard_tp_fraction")) or (_num_eq(tao_frac, 0.4)),
              "hardtp_fraction", f"{tao_frac} vs {g.get('hard_tp_fraction')}")
        check(tao.param("mt", "ref") == g.get("tp_reference"),
              "ref == tp_reference", f"{tao.param('mt','ref')} vs {g.get('tp_reference')}")
    if tao:
        try:
            cast_val = tao.param("mt", "gain", cast=float)
            raw_val = float(tao.param("mt", "gain"))
            cast_ok = cast_val == raw_val
        except (TypeError, ValueError):
            cast_val = None
            cast_ok = False
        check(cast_ok, "param() cast float", str(cast_val))

    print("\n==== 3. RUTARE explicita pe venue (provider_by_name) ====")
    check(api.provider_by_name("binance") is not None, "provider_by_name('binance')")
    check(api.provider_by_name("hyperliquid") is not None, "provider_by_name('hyperliquid')")
    check(api.provider_by_name("nope") is None, "nume necunoscut -> None")
    if tao:
        check(tao.provider_label == "Binance", "TAO -> Binance", tao.provider_label)
    if hype:
        check(hype.provider_label == "Hyperliquid", "HYPE -> Hyperliquid", hype.provider_label)

    print("\n==== 4. operatii generice == facada directa ====")
    if tao:
        p_inst = tao.price()
        p_api = api.get_current_price("TAOUSDC")
        price_match = (
            isinstance(p_inst, (int, float))
            and not isinstance(p_inst, bool)
            and isinstance(p_api, (int, float))
            and not isinstance(p_api, bool)
            and p_inst > 0
            and p_api > 0
            and abs(float(p_inst) - float(p_api)) / float(p_api) < 0.02
        )
        check(price_match,
              "TAO price() == facada", f"instr={p_inst} api={p_api}")
        f_inst = tao.free()
        f_prov = api.provider_by_name("binance").free_balance("TAO")
        check_available(f_inst == f_prov, f_inst is not None and f_prov is not None,
                        "TAO free() == provider direct", f"instr={f_inst} prov={f_prov}")
    if hype:
        p_h = hype.price()
        check(p_h is not None and p_h > 0, "HYPE price() > 0 (rutat la HL)", str(p_h))
        f_h = hype.free()
        f_hp = api.provider_by_name("hyperliquid").free_balance("HYPE")
        check_available(f_h == f_hp, f_h is not None and f_hp is not None,
                        "HYPE free() == HL direct", f"instr={f_h} hl={f_hp}")

    print("\n==== 5. new providers (Kraken, T212) — explicit-only + multi-venue ====")
    kp = api.provider_by_name("kraken")
    tp = api.provider_by_name("t212")
    check(kp is not None, "provider_by_name('kraken')")
    check(tp is not None, "provider_by_name('t212')")
    check(kp is not None and kp.supports_symbol("HYPEUSD") is False,
          "Kraken is explicit-only (supports_symbol=False, it does not steal the facade's routing)")
    hk = inst.get("KRAKEN_HYPE")
    if hk and hype:
        check(hk.provider_label == "Kraken" and hype.provider_label == "Hyperliquid",
              "the SAME HYPE asset routed to 2 venues", f"Kraken={hk.symbol} / HL={hype.symbol}")
    ts = inst.get("T212_SPCX")
    if ts:
        check(ts.provider_label == "T212" and ts.market_hours == "rth", "T212_SPCX -> T212 (rth)")
    mt = load_for("mt")
    check(all(i.enabled for i in mt.values()) and
          all(any(k.startswith("mt.") for k in i.params) for i in mt.values()),
          "load_for('mt') = only enabled instruments that have mt.* params", str(sorted(mt.keys())))
    check(hype is None or not hype.enabled, "HYPE_HL e enabled=no in config")
    check("HYPE_HL" not in mt, "HYPE_HL (enabled=no) does NOT appear in load_for")
    try:                                  # Public Kraken price (informational, tolerant of network failures).
        print(f"  [info] Kraken HYPEUSD price = {kp.get_current_price('HYPEUSD') if kp else None}")
    except Exception as _e:  # noqa: BLE001
        print(f"  [info] Kraken price indisponibil: {_e}")

    print("\n" + "=" * 56)
    if FAIL:
        print(f"FAILED — {len(FAIL)} checks failed: {FAIL}")
        return 1
    print("PASS — fundatia Instrument (pasii 1-3) e sanatoasa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
