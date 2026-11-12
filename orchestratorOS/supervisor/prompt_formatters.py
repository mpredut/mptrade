"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.
Section-level and line-level formatters live in `orchestratorOS.supervisor.prompt_sections`.
"""

from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

from botcore import safe_pos_float, topic_spec
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
        context_items.extend(_format_microstructure_section(microstructure, prices_and_trends))
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
        raw_focus = [focus_symbols] if isinstance(focus_symbols, str) else focus_symbols
        clean_syms = [s.strip().upper() for s in raw_focus if isinstance(s, str) and s.strip()]
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
            f"1. Audit functional system anomalies: fleet processes and modules (procs.conf), bot heartbeats, Linux OS resources (CPU load, RAM, disk), internet & VPN tunnel (wgpia0), configuration files ({cfg_list_str}), and persistent unresolved errors in logs: explain the technical root cause of any bot exceptions, fail-closed blocks, stale heartbeats, OS/network/VPN issues, or configuration errors.",
            "2. Determine whether hung/errored bots or modules require restart (RESTART_BOT / RESTART_MODULE), whether the entire fleet or VPN/DNS requires restart (RESTART_FLEET / RESTART_VPN / PIN_DNS), or whether a critical unrecoverable Linux OS / kernel / system-wide hang requires a full machine reboot (REBOOT_SYSTEM).",
            "3. If a reproducible code bug or configuration error exists in the workspace, formulate an exact search_block and replace_block in `code_remediation` to fix the code or configuration file.",
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
    if spec.can_trade:
        is_equities = getattr(spec, "asset_scope", "all") == "equities"
        target_desc = (
            "ALL live tradable US equity shares on 't212' (SPCX_US_EQ, NVDA_US_EQ, RGNT_US_EQ during US Regular Trading Hours, cross-referencing Kraken read-only xStocks SPCXx.T exposure)"
            if is_equities
            else "ALL live tradable crypto spot coins (BTCUSDC, TAOUSDC, ARBUSDC on 'binance'; HYPEUSDC intensively on 'hyperliquid'; HYPEUSD, TAOUSD, ADAUSD on 'kraken')"
        )
        venue_schema = '"t212"' if is_equities else '"binance" | "kraken" | "hyperliquid"'
        ex_sym = '"NVDA_US_EQ"' if is_equities else '"BTCUSDC"'
        tasks.append(
            f"{len(tasks) + 1}. Formulate symmetric autonomous trading decisions across {target_desc} by synthesizing the "
            "entire MPTrade system telemetry as a whole:\n"
            "   - FULL ACCESS TO EVERYTHING IN `cachedb/` AS A DECISION INFORMATION SOURCE: The telemetry above is a compact "
            "synthesized dashboard (zero raw file dumps); whenever you want deeper detail, you have direct, unrestricted read access "
            "to the entire `cachedb/` (`cachedb/prices/`, `cachedb/trends/`, `cachedb/trends/24trend/`, `cachedb/telemetry/`, "
            "`cachedb/macro/`, `cachedb/orders/`, `cachedb/state/`), `logger/execution_audit/`, `logs/`, and `providers.market_api` "
            "to inspect any file or query any venue on your own.\n"
            "   - BUSINESS OBJECTIVE & FULL AUTONOMOUS JUDGMENT (ZERO PROFIT OR STRATEGY GUARDS): Your objective "
            "is to maximize net portfolio value and risk-adjusted returns (net of venue fees) across all platforms while "
            "protecting capital. ALL deterministic order_guard gates (G_PROF profit reference, GO1 daily cap/cooldown, "
            "GT_* trend deferrals, G_QLIMIT* weight caps, GI_* microstructure, and GLLM* secondary vetoes) are BYPASSED "
            "symmetrically on both BUY and SELL as binding constraints. Deterministic bots (`tradeall.py`, `monitortrades.py`, `assetguardian.py`, "
            "`rtrade.py`, `kraken/bot.py`, `hyperliquid/bot.py`, `212trading/bot.py`) and rigid code filters/configs "
            "(`order_guard.py`, `order_guard.conf`, `instruments.conf`, `intelligence/`) continue working in parallel in the "
            "system and can be consulted by you on demand as advisory indicators and reference guidance—you have full quantitative authority "
            "to validate, adapt, or override them according to live market structure, with ZERO obligation to consult or follow them.\n"
            "   - SUPERVISORY AUTHORITY TO AUDIT & REPAIR / REVERSE UNWANTED BOT OR FLEET TRADES (`by=BOT` vs `by=AI`): "
            "Section 3C explicitly tags every recent fill (`Last Action`, `Last BUY`, `Last SELL`) and every resting order in "
            "`open_limit_orders` with `by=BOT` / `\"actor\": \"BOT\"` (executed by deterministic fleet bots) vs `by=AI` / `\"actor\": \"AI\"` "
            "(executed by you). Actively audit any interim `by=BOT` transaction or resting order executed between your cycles: if a bot or "
            "fleet process made a trade that you disagree with based on your holistic multi-horizon/microstructure/macro view, you have full "
            "authority and duty to immediately repair or reverse it:\n"
            "     1) If a bot SOLD prematurely or when it should NOT have (e.g. panic stop-loss on a liquidity wick, premature micro-take-profit, "
            "or selling at support when absorption/trend structure warrants holding), immediately submit a corrective `BUY` (`MARKET` or `LIMIT`) "
            "to buy back and restore the position.\n"
            "     2) If a bot BOUGHT prematurely or when it should NOT have (e.g. catching a falling knife without bid absorption, buying into "
            "resistance/breakdown, or over-allocating when cash should be preserved), immediately submit a corrective `SELL` (`MARKET` or `LIMIT`) "
            "to liquidate or reduce the unwanted position (all profit/cooldown guards `G_PROF`/`GO1` are bypassed so you can exit immediately "
            "even at a scratch or small loss to repair a bad bot entry).\n"
            "     3) If a bot placed a resting order in `open_limit_orders` (`\"actor\": \"BOT\"`) that you disagree with or that locks wallet "
            "liquidity needed for your repair trade, set `\"cancel_bot_open_orders\": true` (alongside `\"replace_open_orders\": true` if also replacing a prior AI order) to cancel "
            "the bot's resting order and unlock the balance.\n"
            "   - UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING: No rigid trading rules, fixed percentage thresholds, "
            "or directional biases are imposed on you—judge 100% freely on your own WHEN, WHAT, HOW MUCH, on WHICH platform "
            "(`binance`, `kraken`, `hyperliquid`, `t212`), and WHY to `BUY`, `SELL`, or `HOLD`. Historical cost basis (`avg_buy_px`, "
            "`lifo_rem_avg_px`, `open_lifo_lots`) is informational accounting, NEVER a directional constraint on either `SELL` or `BUY`: "
            "when a position or its recent LIFO tranche (`open_lifo_lots`) is in profit (e.g. above `last_buy_px` / `tactical_avg_px` / "
            "`active_avg_buy_px` or after a strong upward swing), you are completely free to harvest the profitable tranche or sell a large "
            "quantity / the entire position (`50%–100%` of `value_usd`, up to "
            f"{max_usd:g} USD) at once without being anchored by older cycle averages; symmetrically, if live price velocity and acceleration "
            "(`kinematics(vel_90s, acc_90s, vel_5m, acc_5m, kalman_z)`), derivatives/spot order flow (`cvd_5m`, `cvd_15m`, `spot_cvd_15m`, "
            "`CB_Premium`), or market structure convince you that a deeper breakdown is unfolding, you are equally free to proactively trim "
            "or exit spot holdings (`MARKET` or `LIMIT`, even at breakeven or below cost basis) to protect NAV and raise quote liquidity to "
            "reload lower—or to hold through noise when data shows genuine absorption.\n"
            "   - CONTRARIAN DIP ACCUMULATION (SCALE-IN) & SYMMETRIC KINEMATIC / BREAKDOWN EVALUATION (`bottom_6h`, `kinematics`): Elevated "
            "macro/geopolitical risk scores, extreme fear, or compressed breadth (`advance_ratio < 0.20`) signal "
            "high-volatility conditions that can produce either high-EV mean-reversion capitulation wicks OR multi-leg structural breakdowns. "
            "Evaluate both possibilities purely from the raw numbers without directional bias and never freeze into passive `HOLD` merely "
            "because an earlier tranche was bought higher: weigh instantaneous velocity and acceleration (`kinematics` in `%/m` and `%/m^2`, "
            "`kalman_vel`, `slopes`) alongside `bottom_6h(prior_low, 1h_low, hl_diff/reclaim)`, `vs_btc`, `cvd_5m`/`cvd_15m`, `spot_cvd_15m`, "
            "`taker_ratio`, `ob_imbalance`, and `bid_wall`/`ask_wall` depth. When downside deceleration (`acc > 0` after a drop) and real "
            "absorption confirm a sweep reclaim or higher low with dry powder available, you may scale in (`MARKET` or `BUY LIMIT`); "
            "conversely, when downward velocity and negative acceleration (`vel < 0`, `acc < 0`) align with unabsorbed spot/perp selling "
            "through key levels, you may proactively de-risk (`SELL`) or stand aside until deceleration confirms.\n"
            "   - DUAL EXECUTION MODE — IMMEDIATE 'MARKET' ORDERS OR PASSIVE 'LIMIT' ORDERS IN THE ORDERBOOK (`open_limit_orders` & `last_cycle_execution_status`): "
            "Because intra-cycle wicks between evaluations can touch key support or resistance levels and bounce before the next cycle, "
            "you may submit either immediate market orders (`\"order_type\": \"MARKET\"`, `\"limit_price\": null`) OR passive resting "
            "limit orders (`\"order_type\": \"LIMIT\"`, `\"limit_price\": <float>`, `\"replace_open_orders\": true`) in the exchange "
            "orderbook (e.g. resting a passive `BUY LIMIT` at a major `bid_wall` / support level below spot with $200–$500 tranches, "
            "or a passive `SELL LIMIT` at an `ask_wall` / take-profit target above spot). Section 3C explicitly reports your active resting orders "
            "(`open_limit_orders`, including `filled_qty`, `remaining_qty`, `status`, `dist_from_spot_pct`, `fill_prob_1h_pct`, and orderbook walls ahead) and mathematical fill confirmation from the previous cycle "
            "(`last_cycle_execution_status` with `status`, `fill_px`, `ref_px`, `filled_qty`, `slippage_bps`), plus the 1-line US Equities/Futures/Macro "
            "correlation index (`SPY / QQQ / ES=F / NQ=F / DXY / US10Y` 24h/1h deltas). Setting `\"replace_open_orders\": true` "
            "automatically cancels any prior unfilled AI limit order on the same symbol and side before placing the updated order; "
            "returning `'HOLD'` leaves existing resting open orders untouched in the book. "
            "ACTIVE ORDERBOOK & RESTING LIMIT ORDER MANAGEMENT (`replace_open_orders: true`): Actively audit every resting order in `open_limit_orders` each cycle—if "
            "a resting `SELL LIMIT` or `BUY LIMIT` is stale/far from spot (`fill_prob_1h_pct <= 5.0%` or `abs(dist_from_spot_pct) > 4.0%`) while the market is ranging "
            "or a profitable LIFO tranche (`profitable_lifo_usd`) can already be harvested near current resistance (`ask_wall` / `LVN` / `VAH` / `POC`), OR if your order "
            "is trapped behind a large `ask_wall_ahead_px` / `bid_wall_ahead_px` (`recommended_front_run_px`), proactively re-anchor it by submitting an updated `LIMIT` "
            "(or `MARKET`) order with `\"replace_open_orders\": true` (e.g. front-running the wall at `recommended_front_run_px` or tightening a far-OTM target to capture "
            "high-probability range rotation rather than leaving capital locked at 1% fill probability for 10–30 hours).\n"
            "   - VOLUME PROFILE (`vol_profile` IN SECTION 3A), LIQUIDATION CLUSTERS (`liq_clusters` IN SECTION 3B) & VENUE FEE/LIMITS "
            "(`venue_fee_per_leg`, `maker`/`taker`, `spread`, `limits` IN SECTION 3C): Reason directly from the raw quantitative numbers without relying on pre-baked labels—use Session Volume Profile "
            "(`POC`, `sec_POC`, `LVN`, 70% Value Area `[VAL, VAH]`, `%` distances, and `va_pos`) "
            "to distinguish genuine breakouts/breakdowns from value-area retests or mean-reversion rotations toward `POC`/`VAL`/`VAH`; "
            "evaluate symmetric `LONG` (downside `long_liq`/`deep_long`) and `SHORT` (upside `short_liq`/`deep_short`) liquidation cluster prices, USD notionals, and `%` distances "
            "to identify where leveraged liquidity pools sit on both sides of spot; and weigh each venue's active `tier`, `maker` vs `taker` fee bps, "
            "live bid-ask `spread` bps, and `limits(min_notional, min_qty, step, tick, max_buy, max_sell)` across `binance` (10/10 bps), "
            "`kraken` (16/26 bps), `hyperliquid` (4/7 bps), and `t212` (15/15 bps) when choosing between aggressive `MARKET` and passive "
            "`LIMIT` execution and sizing `suggested_notional_usd` / `limit_price` cleanly within `[min_notional, max_buy/max_sell]`.\n"
            "   - CROSS-ASSET MACRO, VIX & INSTITUTIONAL CRYPTO/OPTIONS SURFACE SYMMETRY (`SPY / QQQ / VIX / DXY / US10Y`, `Futures_Lead`, `IBIT`, `CB_Premium`, `BTC_DVOL / ETH_DVOL`, `Options_Surface & ETH_Options(VRP, PC_OI, Skew_25d)` IN SECTION 1): "
            "Synthesize US Equities & Rates alongside institutional Crypto & Deribit Options telemetry symmetrically across both Crypto (`binance`, `kraken`, `hyperliquid`) and US Shares (`t212`) directly from the raw numbers: "
            "infer whether global risk appetite is coupled (`COUPLED_RISK_ON` / `COUPLED_RISK_OFF`, where Nasdaq/SPY, VIX, and Crypto move together) or decoupling (`CRYPTO_BULLISH_DECOUPLING` "
            "when Crypto + `IBIT`/`CB_Premium` are bid while Stocks lag, vs `EQUITY_BULLISH_DECOUPLING` when Stocks are bid while Crypto lags); "
            "weigh `Futures_Lead`, `CB_Premium` (US Coinbase vs offshore Binance spot basis bps), `IBIT` (BlackRock ETF delta), `VIX` (S&P 500 implied vol), `BTC_DVOL`/`ETH_DVOL` (30d implied vol), and `Options_Surface`/`ETH_Options(VRP, PC_OI, Skew_25d)` "
            "to distinguish institutional spot accumulation from leveraged derivatives squeezes or macro liquidity flushes.\n"
            f"   - Emit one entry per actionable instrument in `market_trading_decisions` with `confidence >= {min_conf:g}` "
            f"and `suggested_notional_usd` sized between {min_usd:g} and {max_usd:g} USD (the gateway automatically "
            "clamps notional strictly to available free wallet balance with zero artificial cooldowns or dedup locks). "
            "If no instrument warrants action right now, return a single 'HOLD' entry."
        )
        schema.append(
            f'  "market_trading_decisions": [{{"decision": "BUY" | "SELL" | "HOLD", "symbol": {ex_sym}, '
            f'"venue": {venue_schema}, "order_type": "MARKET" | "LIMIT", "limit_price": null, "replace_open_orders": false, "cancel_bot_open_orders": false, '
            f'"confidence": 0.90, "suggested_notional_usd": 1000.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}}]'
        )

    role_intro, tool_rule, report_format_rule = (
        (
            "You are the Lead SRE and Autonomous Technical Repair Supervisor for MPTrade.\n"
            "Your STRICT mandate is solely to detect functional system anomalies (bots, fleet modules, heartbeats), "
            "Linux OS resource issues, internet & VPN (wgpia0) connectivity failures, configuration errors, and "
            "persistent unresolved errors in logs, and to remediate them by patching code or configurations and "
            "restarting/rebooting bots, modules, the trading fleet, VPN/DNS, or the entire Linux system.\n"
            "Do NOT analyze or report on market prices, trends/slopes, macro sentiment, portfolio balances, or active trading positions.\n\n"
            f"{context_str}\n\n"
            "Perform an executive technical diagnostic and self-healing audit:\n" + "\n".join(tasks) + "\n\n",
            "- If you diagnose a reproducible code bug or configuration error and need to verify exact lines before proposing `code_remediation`, "
            "you MAY read the target file in the workspace so `search_block` matches verbatim. "
            "Do NOT edit or modify workspace files directly with tools; propose all fixes exclusively in `code_remediation`.\n",
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** strictly covering technical & SRE health "
            "with the cycle timestamp in the main header (e.g. `### 🛠️ Raport Audit Sistem [HH:MM:SS EET]`) "
            "(bullet points for: Starea Flotei, Boților, Sistemului Linux & VPN (wgpia0) & Loguri, "
            "Conectivitate Internet & Audit Configurații, Erori Persistente Nerezolvate, și "
            "Acțiuni de Remediere Cod/Config & Reboot Boți/Module/Flotă/Sistem). "
            "NU analiza și NU menționa prețuri de piață, pante, portofoliu, solduri sau poziții active!\n",
        )
        if spec.can_self_heal
        else (
            "You are the Quantitative Autonomous Portfolio Manager and Trade Executor for MPTrade.\n"
            "Analyze the complete market telemetry, macro/geopolitical sensors, microstructure, spot wallet balances, and cycle states:\n\n"
            f"{context_str}\n\n"
            "Evaluate the market context and formulate autonomous BUY/SELL trading decisions:\n" + "\n".join(tasks) + "\n\n",
            "- You have full, unrestricted access to everything in `cachedb/` (`cachedb/prices/`, `cachedb/trends/`, `cachedb/trends/24trend/`, "
            "`cachedb/telemetry/`, `cachedb/macro/`, `cachedb/orders/`, `cachedb/state/`) as a direct source of information for your trading decisions, "
            "as well as full freedom to query live market APIs (`providers.market_api`, `binance_api`, `kraken`, `hyperliquid`, `212trading`) "
            "and read trade/execution logs (`logs/`, `logger/execution_audit/`) whenever you want deeper verification or to answer user questions.\n"
            "- Deterministic bots (`tradeall.py`, `monitortrades.py`, `assetguardian.py`, `rtrade.py`, venue bots) and rigid code filters/guards "
            "(`order_guard.py`, `order_guard.conf`, `instruments.conf`, `intelligence/`) run in parallel and may be inspected on demand if you wish, "
            "but consulting or following them is never required; if any bot executed an interim BUY or SELL (`by=BOT`) that you disagree with, "
            "you are fully empowered to immediately repair/reverse it with a counter-trade (`BUY` back what a bot wrongly sold, or `SELL` what a bot wrongly bought).\n"
            "- Keep this thread focused on market analysis, portfolio management, and autonomous BUY/SELL execution across venues.\n",
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** with the cycle timestamp in the main header "
            "(e.g. `### ⏱️ Raport Execuție [HH:MM:SS EET]`), bullet points for: "
            "Piața Live & Pante (Toate Monedele), Portofoliu, Solduri & Poziții Active, și Decizia AI de Tranzacționare / Răspuns la intervențiile din chat).\n",
        )
    )

    return (
        role_intro
        + "INTERACTIVE THREAD RULES:\n"
        "- If the user wrote any tactical instructions, price targets, overrides, or questions earlier in this "
        "conversation thread, honor them with highest priority and address them directly.\n"
        + tool_rule
        + report_format_rule
        + "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
        "```json\n{\n" + ",\n".join(schema) + "\n}\n```"
    )

