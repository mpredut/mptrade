"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.
Section-level and line-level formatters live in `orchestratorOS.supervisor.prompt_sections`.
"""

from datetime import datetime
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from botcore import (
    is_us_rth_open,
    load_key_value_conf,
    parse_symbol_list,
    safe_nonneg_float,
    safe_pos_float,
    topic_spec,
)
from cachepaths import is_kraken_tokenized_asset
from instrument_registry import discover_tradable_specs
from intelligence.macro_sentiment_ssot import (
    CRYPTO_ONLY_MACRO_SENSOR_KEYS,
    format_telemetry_domain_age_tag,
    resolve_telemetry_domain_stale_thresholds_sec,
)
from orchestratorOS.lib.technical_repair_ssot import (
    ALL_DIAGNOSED_CATEGORIES,
    ALL_REPAIR_ACTIONS,
    AUDITED_CONFIG_FILES,
    DIAGNOSED_SEVERITIES,
)
from orchestratorOS.supervisor.prompt_sections import (
    _format_macro_and_news_section,
    _format_market_prices_section,
    _format_microstructure_section,
    _format_portfolio_section,
    _format_sre_and_fleet_section,
)
from providers.ai_circuit_breaker import (
    DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
    DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
    parse_intent_circuit_breaker_config,
)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_xstock_sell_prompt_thresholds(workspace_dir: Optional[str] = None) -> Tuple[float, float]:
    """Read SELL-only tokenized equity profit/drop thresholds from order_guard.conf SSOT."""
    ws = workspace_dir or _ROOT_DIR
    try:
        conf = load_key_value_conf(ws, _ROOT_DIR)
        cb_cfg = parse_intent_circuit_breaker_config(conf)
        return (
            safe_nonneg_float(cb_cfg.get("xstock_sell_min_profit_pct"), default=DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT),
            safe_nonneg_float(
                cb_cfg.get("xstock_sell_min_target_drop_pct"),
                default=DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
            ),
        )
    except Exception:
        return DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT, DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT


def _build_dynamic_target_desc(
    is_equities: bool, *, workspace_dir: Optional[str] = None
) -> Tuple[str, str]:
    """Format active tradable instruments per venue dynamically from instruments.conf SSOT."""
    ws = workspace_dir or _ROOT_DIR
    scope = "equities" if is_equities else "crypto"
    try:
        specs = discover_tradable_specs(ws, asset_scope=scope, include_closed_rth=True)
    except Exception:
        specs = {}
    by_venue: Dict[str, List[str]] = {}
    xstock_syms: List[str] = []
    for item in specs.values():
        prov = str(getattr(item, "provider", "") or "").strip().lower()
        sym = str(getattr(item, "symbol", "") or "").strip()
        if not (prov and sym):
            continue
        is_xs = is_kraken_tokenized_asset(sym)
        if is_xs and sym not in xstock_syms:
            xstock_syms.append(sym)
        lbl = f"{sym} [0bps SELL-only xStock]" if (not is_equities and is_xs) else sym
        bucket = by_venue.setdefault(prov, [])
        if lbl not in bucket:
            bucket.append(lbl)

    if is_equities:
        eq_syms = [s for syms in by_venue.values() for s in syms]
        eq_list = ", ".join(eq_syms) if eq_syms else "enabled T212 US equities"
        return (
            f"ALL live tradable US equity shares on 't212' ({eq_list} during US Regular Trading Hours, "
            "cross-referencing Kraken 24/7 xStock exposure)",
            "xStocks",
        )
    venue_parts = [f"{', '.join(syms)} on '{prov}'" for prov, syms in by_venue.items() if syms]
    target_desc = (
        f"ALL live tradable crypto & 24/7 tokenized xStock pairs ({'; '.join(venue_parts)})"
        if venue_parts
        else "ALL live tradable crypto & 24/7 tokenized xStock pairs configured in instruments.conf"
    )
    xs_lbl = (
        ", ".join(f"`{s}`" for s in xstock_syms)
        if xstock_syms
        else "tokenized xStocks"
    )
    return target_desc, xs_lbl


def build_supervisor_prompt(
    sensors: Any,
    logs: Any,
    spec: Any = None,
    lookback_hours: float = 1.0,
    focus_symbols: Optional[List[str]] = None,
    *,
    min_usd: float = 10.0,
    max_usd: float = 2000.0,
    min_conf: float = 0.85,
    topic_key: Optional[str] = None,
    min_notional_usd: Optional[float] = None,
    max_notional_usd: Optional[float] = None,
    min_confidence: Optional[float] = None,
    compact_rules: bool = False,
    **_extra: Any,
) -> str:
    """Build the topic-specific prompt for Gemini (auto_technical_repair or auto_*trade_executor)."""
    if spec is None:
        spec = topic_spec(topic_key or _extra.get("topic") or "auto_cryptotrade_executor")
    lookback_hours = safe_pos_float(lookback_hours, default=1.0)
    min_usd = safe_pos_float(min_notional_usd if min_notional_usd is not None else min_usd, default=10.0)
    max_usd = safe_pos_float(max_notional_usd if max_notional_usd is not None else max_usd, default=2000.0)
    min_conf = safe_pos_float(min_confidence if min_confidence is not None else min_conf, default=0.85)

    is_pure_sre = bool(spec.can_self_heal and not spec.can_trade)
    now_ts_str = datetime.now(ZoneInfo("Europe/Bucharest")).strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    context_items: List[str] = [
        f"### 0. RUN TIMESTAMP & OPERATIONAL CLOCK: {now_ts_str}\n"
    ]
    domain_ages = getattr(sensors, "source_ages_sec", None)
    eval_now_ts = safe_pos_float(getattr(sensors, "evaluated_at_ts", None), default=None)
    spec_scope_norm = str(getattr(spec, "asset_scope", "all") or "all").strip().lower()
    skip_domain_keys = (
        CRYPTO_ONLY_MACRO_SENSOR_KEYS
        if spec_scope_norm in ("equities", "equity")
        else frozenset()
    )
    if not is_pure_sre and isinstance(domain_ages, dict) and domain_ages:
        eff_now_s0 = eval_now_ts if eval_now_ts is not None else time.time()
        stale_thrs = resolve_telemetry_domain_stale_thresholds_sec(spec_scope_norm, now_ts=eff_now_s0)
        us_rth_open = is_us_rth_open(eff_now_s0)
        age_tokens: List[str] = []
        for k, v in domain_ages.items():
            if k in skip_domain_keys:
                continue
            v_f = safe_pos_float(v, default=None)
            if v_f is None and v != 0.0:
                continue
            stale_tag = format_telemetry_domain_age_tag(
                k,
                v_f or 0.0,
                asset_scope=spec_scope_norm,
                us_rth_open=us_rth_open,
                thresholds=stale_thrs,
            )
            age_tokens.append(f"{k}={v}s{stale_tag}")
        if age_tokens:
            context_items.append(f"- Snapshot Domain Ages: {', '.join(age_tokens)}")
    if not is_pure_sre:
        context_items.extend(_format_macro_and_news_section(sensors, getattr(spec, "asset_scope", "all")))
    if spec.can_self_heal:
        context_items.extend(_format_sre_and_fleet_section(sensors, logs, lookback_hours))
    prices_and_trends = getattr(sensors, "prices_and_trends", None) or {}
    microstructure = getattr(sensors, "microstructure", None) or {}
    portfolio_summary = getattr(sensors, "portfolio_summary", None) or {}
    if not is_pure_sre and prices_and_trends:
        context_items.extend(_format_market_prices_section(prices_and_trends))
    if not is_pure_sre and microstructure:
        context_items.extend(_format_microstructure_section(microstructure, prices_and_trends, now_ts=eval_now_ts))
    if not is_pure_sre and portfolio_summary:
        context_items.extend(
            _format_portfolio_section(
                portfolio_summary,
                getattr(spec, "asset_scope", "all"),
                max_usd=max_usd,
                include_macro_benchmarks=False,
            )
        )
    if not is_pure_sre and focus_symbols:
        clean_syms = parse_symbol_list(focus_symbols)
        if clean_syms:
            context_items.append(
                f"\n### 4. TACTICAL PRIORITY SYMBOLS: {', '.join(clean_syms)}\n"
                "- Pay heightened attention to these symbols alongside all other live tradable coins above."
            )

    context_str = "\n".join(context_items)
    tasks: List[str] = []
    schema: List[str] = []
    if spec.can_self_heal:
        sev_union = " | ".join(f'"{s}"' for s in DIAGNOSED_SEVERITIES)
        cat_union = " | ".join(f'"{c}"' for c in ALL_DIAGNOSED_CATEGORIES)
        act_union = " | ".join(f'"{a}"' for a in ALL_REPAIR_ACTIONS)
        cfg_list_str = ", ".join(AUDITED_CONFIG_FILES)
        tasks.extend([
            f"1. Audit functional system anomalies: fleet processes and modules (procs.conf), bot heartbeats, Linux OS resources (CPU load, RAM, disk), internet & VPN tunnel (wgpia0), configuration files ({cfg_list_str}), LLM model catalog availability (config.env vs agy models), and persistent unresolved errors in logs: explain the technical root cause of any bot exceptions, fail-closed blocks, stale heartbeats, OS/network/VPN issues, or configuration errors.",
            "2. Determine whether hung/errored bots or modules require restart (RESTART_BOT / RESTART_MODULE), whether the entire fleet or VPN/DNS requires restart (RESTART_FLEET / RESTART_VPN / PIN_DNS), or whether a critical unrecoverable Linux OS / kernel / system-wide hang requires a full machine reboot (REBOOT_SYSTEM).",
            "3. If a reproducible code bug or configuration error exists in the workspace, formulate an exact search_block and replace_block in `code_remediation` to fix the code or configuration file. Note: when AUTO_CONFIG_UPDATES or MISSING_CONFIGURED_MODELS are reported for config.env, auto_technical_repair updates config.env automatically via verified code_remediation so the next fleet hot-reload adopts the new models.",
        ])
        schema.extend([
            '  "summary": "1-2 sentence executive technical assessment of functional system, Linux OS, internet/VPN, configurations, and fleet health"',
            f'  "diagnosed_issues": [{{"severity": {sev_union}, '
            f'"category": {cat_union}, '
            '"description": "Short explanation", "root_cause": "Detailed technical root cause"}]',
            f'  "venue_actions": [{{"action": {act_union}, '
            '"venue": "fleet" | "system" | "binance" | "kraken" | "hyperliquid" | "t212", "symbol": "bot_or_module_name or SYSTEM", '
            '"order_id": "optional order id", "reason": "Why this technical repair or restart/reboot action is needed"}]',
            '  "code_remediation": {"has_fix": false, "target_file": "", "rationale": "", '
            '"search_block": "", "replace_block": ""}',
        ])
    else:
        schema.append(
            '  "summary": "1-2 sentence executive assessment of multi-coin market posture and trading decisions"'
        )
    xs_min_profit_pct, xs_min_drop_pct = _resolve_xstock_sell_prompt_thresholds()
    xs_target_lbl = "tokenized xStocks"
    if spec.can_trade:
        is_equities = getattr(spec, "asset_scope", "all") == "equities"
        target_desc, xs_target_lbl = _build_dynamic_target_desc(is_equities)
        venue_schema = '"t212"' if is_equities else '"binance" | "kraken" | "hyperliquid"'
        venue_list_desc = "`t212`" if is_equities else "`binance`, `kraken`, `hyperliquid`"
        ex_sym = '"NVDA_US_EQ"' if is_equities else '"BTCUSDC"'
        tasks.append(
            f"{len(tasks) + 1}. Formulate symmetric autonomous trading decisions across {target_desc}:\n"
            "   - FULL ACCESS TO EVERYTHING IN `cachedb/` AS A DECISION INFORMATION SOURCE: Telemetry below is a compact dashboard; "
            "you have unrestricted read access to `cachedb/` (including `cachedb/orders/` Order Ledger SSoT), `logs/`, and `providers.market_api` for deeper verification.\n"
            "   - ZERO PROFIT OR STRATEGY GUARDS (AI SURVIVAL BREAKERS ONLY): Maximize risk-adjusted net NAV (after venue fees). All deterministic order_guard gates "
            "(G_PROF, GO1, GT_*, G_QLIMIT*, GI_*, GLLM*) are BYPASSED symmetrically on BUY and SELL. Deterministic bots and configs "
            "continue working in parallel in the system and can be consulted by you on demand as advisory references only--ZERO obligation to follow them. "
            "Only AI-specific survival circuit breakers "
            "(anti-ping-pong flip cooldown unless `invalidation_px`/`target_px` or net edge triggers, 6h whipsaw cap, daily `by=AI` BUY notional cap, and 48h `AI Decision Scorecard` BUY shadow cooldown) apply in the gateway.\n"
            "   - SUPERVISORY AUTHORITY TO AUDIT & REPAIR / REVERSE UNWANTED BOT OR FLEET TRADES (`by=BOT` vs `by=AI`): "
            "Audit interim `by=BOT` fills/orders independently. Cancel unwanted unfilled bot orders BEFORE they fill (`cancel_bot_open_orders: true` "
            "with mandatory `cancel_order_ids: [...]` on standalone `HOLD`, or optionally narrowed via `cancel_order_ids` on `BUY`/`SELL`); do NOT sell in the first few minutes after a bot fill unless structure genuinely breaks down "
            "(avoid burning round-trip fees on normal noise). When a real repair is warranted, submit a spot `LIMIT` order (`\"order_type\": \"LIMIT\"`) "
            "priced as close as possible to the current live market price (`best_bid` for BUY / `best_ask` for SELL):\n"
            "     1) If a bot SOLD prematurely or when it should NOT have, submit a corrective `BUY` `LIMIT` right at live spot/`best_bid` to restore at maker fee.\n"
            "     2) If a bot BOUGHT prematurely or when it should NOT have (confirmed structural breakdown), submit a corrective `SELL` `LIMIT` right at live spot/`best_ask`.\n"
            "     3) Targeting specific orders with `cancel_order_ids: [...]` + `cancel_bot_open_orders: true` / `replace_open_orders: true` cancels only those orders "
            "(standalone `HOLD` cancel requires explicit `cancel_order_ids`; on `BUY`/`SELL` without `cancel_order_ids`, only same-side orders cancel; if cancel fails, replacement aborts).\n"
            f"   - UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING: Decide freely WHEN/WHAT/HOW MUCH/WHICH venue ({venue_list_desc}). "
            "Cost basis (`avg_buy_px`, `lifo_rem_avg_px`, `open_lifo_lots`) is informational, NEVER a constraint on 2-way pairs: harvest profitable LIFO tranches or up to `50%\u2013100%` of `value_usd` "
            f"(up to {max_usd:g} USD), OR trim/exit via near-market `LIMIT` even at breakeven or below cost basis on breakdown "
            f"(EXCEPTION: SELL-only Kraken {xs_target_lbl} = long-term `HOLD`, heavy manual rebuy fee/spread; sell rarely ONLY on >+{xs_min_profit_pct:g}% profit + 1-2d+ drop, `target_px` >={xs_min_drop_pct:g}% lower; emailed to rebuy manually).\n"
            "   - CONTRARIAN DIP ACCUMULATION (SCALE-IN) & KINEMATICS: Weigh `kinematics(vel_90s, acc_90s, vel_5m, acc_5m, kalman_z)`, `kalman_vel`, `slopes`, "
            "`bottom_6h`, `vs_btc`, `cvd_5m`/`cvd_15m`, `spot_cvd_15m` (`[real_spot]` observed vs `[est:ob+15m]` proxy\u2014never double-count proxy alongside `ob_imbalance`), "
            "`spot_vol_15m`/`spot_vol_1h` & `rvol_15m`/`rvol_1h` (volume expansion vs noise), `taker_ratio`, and L2 walls. Scale in on confirmed downside deceleration + absorption; "
            "de-risk when negative velocity/acceleration aligns with unabsorbed selling.\n"
            "   - DUAL EXECUTION MODE \u2014 IMMEDIATE 'MARKET' ORDERS OR PASSIVE 'LIMIT' ORDERS IN THE ORDERBOOK: Prefer `\"order_type\": \"LIMIT\"` for low maker fees "
            "(passive `limit_price` at bid/ask walls for swings; right at spot/BBO for urgent entries/reversals, auto-stepped 1 tick onto maker book). "
            "Audit `open_limit_orders` (`dist_from_spot_pct` & `fill_prob_1h_pct` anchored to live spot; watch `[ob_stale]` when `ob_age > 30s`) and "
            "`last_cycle_execution_status` (emitted for recent `<=60m` fill/rejection or `<=30m` resting order): re-anchor stale/far (`fill_prob_1h_pct <= 5%` or `|dist| > 4%`) "
            "or wall-trapped orders (`recommended_front_run_px`) via `replace_open_orders: true`; returning `'HOLD'` leaves resting orders untouched.\n"
            "   - STRUCTURE, LIQUIDATIONS & CROSS-ASSET MACRO, VIX & INSTITUTIONAL CRYPTO/OPTIONS SURFACE SYMMETRY: Combine `vol_profile` (`POC`/`VAL`/`VAH`/`LVN`, `REAL_VOL` vs `TPO`), "
            "`liq_clusters` (`src=model_est` heuristic bands; prioritize `realized_liq_15m` when present), `book_window` cap tags, per-venue fees/spread/`limits`, and Section 1 cross-asset telemetry "
            "(`SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y`, `Futures_Lead`, `IBIT`, `CB_Premium`, `BTC_DVOL/ETH_DVOL`, `Options_Surface` & `ETH_Options`, judging coupled risk-on/off vs cross-asset decoupling directly from the numbers; positive `Skew_25d` = Put_IV - Call_IV > 0 reflects put-buying fear, negative reflects call demand).\n"
            f"   - Emit one entry per actionable instrument in `market_trading_decisions` (`confidence >= {min_conf:g}`, `suggested_notional_usd` between {min_usd:g} and {max_usd:g} USD; "
            "gateway clamps strictly to free balance with zero artificial cooldowns or dedup locks beyond AI survival breakers), or a single `'HOLD'` entry. "
            "On actionable BUY/SELL, set numeric `invalidation_px` (thesis break) and `target_px` (primary objective); active plans persist in Section 3C across compactions."
        )
        schema.append(
            f'  "market_trading_decisions": [{{"decision": "BUY" | "SELL" | "HOLD", "symbol": {ex_sym}, '
            f'"venue": {venue_schema}, "order_type": "MARKET" | "LIMIT", "limit_price": null, '
            f'"invalidation_px": null, "target_px": null, '
            f'"replace_open_orders": false, "cancel_bot_open_orders": false, "cancel_order_ids": [], '
            f'"confidence": 0.90, "suggested_notional_usd": 1000.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}}]'
        )

    if compact_rules and not spec.can_self_heal:
        compact_prefix = (
            "CONTINUATION TURN (ACTIVE THREAD POLICY APPLIES):\n"
            "- Apply all quantitative trading, spot maker-LIMIT execution, active orderbook management, and "
            "`by=BOT` vs `by=AI` supervisory rules established earlier in this thread (unrestricted read access to "
            f"`cachedb/`, zero deterministic guards, `confidence >= {min_conf:g}`, notional {min_usd:g}-{max_usd:g} USD; "
            f"remember SELL-only Kraken {xs_target_lbl} = long-term HOLD with heavy manual rebuy fee: sell rarely ONLY on >+{xs_min_profit_pct:g}% profit AND 1-2+ day downside with `target_px` >={xs_min_drop_pct:g}% lower).\n"
            "- Honor any user chat instructions, price targets, overrides, or questions earlier in this thread with highest priority.\n"
            "- Format your response in TWO parts:\n"
            "  1) A concise Markdown report in **Romanian** with the cycle timestamp header (`### \u23f1\ufe0f Raport Execu\u021bie [HH:MM:SS EET]`): "
            "on routine pure `HOLD` cycles (no BUY/SELL/cancel, no Section 4 priority symbols, no user chat message), output ONLY "
            "2-3 ultra-compact bullets (NAV + free cash, 1-line market pulse, HOLD rationale; omit coin-by-coin tables); "
            "when executing BUY/SELL/cancel, evaluating Section 4 priority symbols, or replying to user chat, provide the full "
            "concise coin-by-coin Romanian breakdown.\n"
            "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
            "```json\n{\n" + ",\n".join(schema) + "\n}\n```\n\n"
            "LIVE TELEMETRY SNAPSHOT:\n"
        )
        return compact_prefix + context_str

    role_intro, tool_rule, report_format_rule = (
        (
            "You are the Lead SRE and Autonomous Technical Repair Supervisor for MPTrade.\n"
            "Your STRICT mandate is solely to detect functional system anomalies (bots, fleet modules, heartbeats), "
            "Linux OS resource issues, internet & VPN (wgpia0) connectivity failures, configuration errors, and "
            "persistent unresolved errors in logs, and to remediate them by patching code or configurations and "
            "restarting/rebooting bots, modules, the trading fleet, VPN/DNS, or the entire Linux system.\n"
            "Do NOT analyze or report on market prices, trends/slopes, macro sentiment, portfolio balances, or active trading positions.\n\n"
            "Perform an executive technical diagnostic and self-healing audit based on the live telemetry snapshot below:\n"
            + "\n".join(tasks)
            + "\n\n",
            "- If you diagnose a reproducible code bug or configuration error and need to verify exact lines before proposing `code_remediation`, "
            "you MAY read the target file in the workspace so `search_block` matches verbatim. "
            "Do NOT edit or modify workspace files directly with tools; propose all fixes exclusively in `code_remediation`.\n",
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** strictly covering technical & SRE health "
            "with the cycle timestamp in the main header (e.g. `### \U0001f6e0\ufe0f Raport Audit Sistem [HH:MM:SS EET]`) "
            "(bullet points for: Starea Flotei, Bo\u021bilor, Sistemului Linux & VPN (wgpia0) & Loguri, "
            "Conectivitate Internet & Audit Configura\u021bii, Erori Persistente Nerezolvate, \u0219i "
            "Ac\u021biuni de Remediere Cod/Config & Reboot Bo\u021bi/Module/Flot\u0103/Sistem). "
            "NU analiza \u0219i NU men\u021biona pre\u021buri de pia\u021b\u0103, pante, portofoliu, solduri sau pozi\u021bii active!\n",
        )
        if spec.can_self_heal
        else (
            "You are the Quantitative Autonomous Portfolio Manager and Trade Executor for MPTrade.\n"
            "Evaluate the telemetry snapshot below and formulate autonomous BUY/SELL trading decisions:\n"
            + "\n".join(tasks)
            + "\n\n",
            "- Keep this thread focused on market analysis, portfolio management, and autonomous BUY/SELL execution across venues.\n",
            "- Format your response in TWO parts:\n"
            "  1) A concise Markdown report in **Romanian** with cycle timestamp header (`### \u23f1\ufe0f Raport Execu\u021bie [HH:MM:SS EET]`):\n"
            "     - On routine pure `HOLD` cycles (no BUY/SELL/cancel action, no Section 4 priority symbols, no user chat message), "
            "keep the report ULTRA-COMPACT (2-3 short bullet lines max: NAV + free cash, 1-line market pulse, HOLD rationale; omit coin tables).\n"
            "     - On BUY/SELL/cancel, Section 4 priority symbols, or user chat replies, include concise bullet points for: "
            "Pia\u021ba Live & Pante (Toate Monedele), Portofoliu, Solduri & Pozi\u021bii Active, \u0219i Decizia AI de Tranzac\u021bionare / R\u0103spuns la interven\u021biile din chat).\n",
        )
    )

    static_prefix = (
        role_intro
        + "INTERACTIVE THREAD RULES:\n"
        "- If the user wrote any tactical instructions, price targets, overrides, or questions earlier in this "
        "conversation thread, honor them with highest priority and address them directly.\n"
        + tool_rule
        + report_format_rule
        + "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
        "```json\n{\n" + ",\n".join(schema) + "\n}\n```\n\n"
        "LIVE TELEMETRY SNAPSHOT:\n"
    )
    return static_prefix + context_str

