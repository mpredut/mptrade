# Runtime state ownership and locking

Atomic publication prevents partial files; it does not serialize a complete
read-modify-write transaction. This inventory records the separate ownership rule.

| State | Writer ownership | Serialization | Assessment |
|---|---|---|---|
| Strategy state (`strategies/state_store.py`) | one bot per venue/instrument | launcher `single_instance` | safe for normal fleet launch |
| AssetGuardian tranches | AssetGuardian operations | `FileLock` around read and write | cross-process serialized |
| rtrade pair rounds | rtrade operations | `FileLock` around mutation | cross-process serialized |
| retry outbox | all execution processes | dedicated queue lock plus file/directory sync | intentionally specialized |
| Cross-venue Order Ledger SSoT (`order_events_YYYY-MM-DD.jsonl`, `orders_active.json`, `orders_closed_YYYY-MM.jsonl`) | multi-producer append-only events (`OrderLedgerWriter`); single-writer materialized active/closed files (`order_ledger_reconciler_worker.py::OrderLedgerReconciler`, 100% network-free) | `flock` on JSONL appends; `generation_file_lock` + `save_json_state` on `orders_active.json`; passive `OrderLedgerReader` keeps `repair_files=False` with in-RAM tail overlay | cross-process serialized; strict SoC vs `cacheManager` |
| Hyperliquid delta-neutral | one process per coin | state-specific `flock` | cross-process serialized |
| Binance/Kraken trailing | daemon plus optional one-shot command | daemon and `--once` share `single_instance`; status is read-only | serialized |
| Kraken xStock watcher | daemon plus optional one-shot command | daemon and `--once` share `single_instance`; trial state is isolated | serialized |
| Hyperliquid perpetual strategy | one process per coin | coin-specific `single_instance`, including test strategy | serialized through launcher |
| Kraken strategy | one process per pair | pair-specific `single_instance`, including test strategy | serialized through launcher |
| Price, equity OHLCV & short/Kalman trend caches (`cache_current_price.json`, `cache_24_price_<SYM>.json`, `cache_ohlc_24h_<SYM>.json`, `cache_short_trend.json`, `cache_kalman_trend.json`, `24trend/live_*.json`) | `cacheManager.py` (`MODE_PRODUCER`; stitched 24h fallback in `price_series_evaluator.py` under `generation_file_lock` without overwriting fresh `< 240s` equity OHLCV) | `single_instance` + `.generation.lock` (`commit_file_and_meta` / `generation_file_lock`); `tradeall.py` runs `KalmanStateTracker(read_only=True)` in memory; `212trading/market_data.py` reads SSOT equity price & 5m OHLCV first before Yahoo HTTP fallback; consumers use `MODE_LIVE_READER` / `MODE_DISK_READER` with `repair_files=False` | single producer; passive readers never mutate `.previous` or `cache_kalman_trend.json` |
| Macro, breadth, pre-trade & venue balance telemetry (`market_breadth_collect.json`, `fear_greed_collect.json`, `news_feed_collect.json`, `venue_balances_state.json`, `microstructure_cycle_history.json`) | `intelligence/globaltelemetry_collector.py` (`portfolio_collector.py` + `microstructure_collector.py`) | single periodic collector daemon; `venue_balances_state.json` merges under `generation_file_lock` across portfolio & T212 position-price updates; `AutonomousAIReconciler` reads passively via `read_snapshot(allow_network=False, persist=False)` | serialized merge; supervisor never overwrites telemetry caches |
| Trend survival empirical cache (`trend_survival_cache.json`) | `intelligence/internal/state/survival.py` (`estimate_T`) | `generation_file_lock` (`.generation.lock`) across post-fetch disk re-read and per-symbol merge | cross-process serialized; sibling symbols preserved |
| Autonomous trade intents (`autonomous_trade_intents_state.json`) | `AutonomousIntentGateway` + reconcilers | `durable_intents_lock` (`FileLock`) across read-modify-write | cross-process serialized |
| Venue JSON trade & order caches (`cache_trade_kraken.json`, `cache_trade_hyperliquid.json`, `cache_trade_t212.json`, `cache_order_filled_<venue>.json`, `cache_order_open_<venue>.json`) | `cacheManager` WS/Ledgers/OpenOrders bridges + `portfolio_collector` live open-order sync | `generation_file_lock` (`.generation.lock`) across on-disk merge, replace, touch, and invalidation; `venue_balances_state.json` stores balances & positions only | cross-process serialized; no stale RAM snapshot overwrite |
| Notification delivery & retry queue (`statedb/system/notification_delivery_state.json`, `statedb/system/notify_queue.json`) | `NotificationDeliveryStateStore` + `NotifyQueueStore` (`notify_engine/alertnotifiers.py` + `NotificationServer`) | `generation_file_lock` across gate updates, `.previous` repair, enqueue, and retry flush | cross-process serialized |
| OS resource watchdog (`statedb/system/os_resources_watchdog_state.json`) | `OsResourcesWatchdogStateStore` (`orchestratorOS/livecheck/os_resources_watchdog.py`) | cron single-writer atomic `save_json_state` + passive read in `technical_repair_ssot.py` | serialized |
| Smart stop-loss grace state (`smart_stop_loss_state.json`) | `SmartStopLossGuard` across monitoring/strategy bots | `generation_file_lock` (`.generation.lock`) with per-symbol disk merge/remove | cross-process serialized; sibling symbols preserved |
| LLM topic thread registry (`llm_topic_threads.json`) | `orchestratorOS/agy/topics.py` across macro, guard, and executor topics | `generation_file_lock` (`.generation.lock`) across per-topic session rotation, turn recording, and title sync | cross-process serialized; sibling topics preserved |
| Cache managers (account/archive) | one fleet process plus internal threads | fleet ownership and in-process/generation locks | safe while deployment preserves a single process |
| Shadow/monitoring state | cron-owned | atomic last-writer-wins | acceptable; no financial authorization |

## Follow-up decisions

Do not add a lock inside `state_io` atomic write helpers: doing so would protect only one
write, not the read-modify-write transaction. The owning component must hold a lock across
its full mutation, and passive readers (`read_json_file`, `read_json_dict`,
`read_json_recovering`, `read_json_dict_recovering`, and `CacheManagerInterface` in
`read_only` mode) must keep `repair_files=False` so `.previous` promotion and cleanup
happen strictly inside `MODE_PRODUCER` under `.generation.lock`. The formerly identified
one-shot, strategy-launch, and supervisor telemetry-write overlaps are closed. Retain
deployment and policy tests that guarantee single-producer cache ownership and passive
supervisor snapshot reads. Direct library construction outside these launchers remains the
caller's responsibility.
