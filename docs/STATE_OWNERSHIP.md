# Runtime state ownership and locking

Atomic publication prevents partial files; it does not serialize a complete
read-modify-write transaction. This inventory records the separate ownership rule.

| State | Writer ownership | Serialization | Assessment |
|---|---|---|---|
| Strategy state (`strategies/state_store.py`) | one bot per venue/instrument | launcher `single_instance` | safe for normal fleet launch |
| AssetGuardian tranches | AssetGuardian operations | `FileLock` around read and write | cross-process serialized |
| rtrade pair rounds | rtrade operations | `FileLock` around mutation | cross-process serialized |
| retry outbox | all execution processes | dedicated queue lock plus file/directory sync | intentionally specialized |
| Hyperliquid delta-neutral | one process per coin | state-specific `flock` | cross-process serialized |
| Binance/Kraken trailing | daemon plus optional one-shot command | daemon and `--once` share `single_instance`; status is read-only | serialized |
| Kraken xStock watcher | daemon plus optional one-shot command | daemon and `--once` share `single_instance`; trial state is isolated | serialized |
| Hyperliquid perpetual strategy | one process per coin | coin-specific `single_instance`, including test strategy | serialized through launcher |
| Kraken strategy | one process per pair | pair-specific `single_instance`, including test strategy | serialized through launcher |
| Price, equity OHLCV & short/Kalman trend caches (`cache_current_price.json`, `cache_24_price_<SYM>.json`, `cache_ohlc_24h_<SYM>.json`, `cache_short_trend.json`, `cache_kalman_trend.json`, `24trend/live_*.json`) | `cacheManager.py` (`MODE_PRODUCER`) | `single_instance` + `.generation.lock` (`commit_file_and_meta`); `tradeall.py` runs `KalmanStateTracker(read_only=True)` in memory; consumers use `MODE_LIVE_READER` / `MODE_DISK_READER` with `repair_files=False` | single producer; passive readers never mutate `.previous` or `cache_kalman_trend.json` |
| Macro, breadth, pre-trade & venue balance telemetry (`market_breadth_collect.json`, `fear_greed_collect.json`, `news_feed_collect.json`, `venue_balances_state.json`, `microstructure_cycle_history.json`) | `intelligence/globaltelemetry_collector.py` | single periodic collector daemon; `AutonomousAIReconciler` reads passively via `read_snapshot(allow_network=False, persist=False)` | single producer; supervisor never overwrites telemetry caches |
| Autonomous trade intents (`autonomous_trade_intents_state.json`) | `AutonomousIntentGateway` + reconcilers | `durable_intents_lock` (`FileLock`) across read-modify-write | cross-process serialized |
| Venue JSON trade caches (`cache_trade_kraken.json`, `cache_trade_hyperliquid.json`, `cache_trade_t212.json`) | `cacheManager` WS/Ledgers bridges + provider fill sync/invalidation | `generation_file_lock` (`.generation.lock`) across on-disk merge, touch, and invalidation | cross-process serialized; no stale RAM snapshot overwrite |
| Notification delivery & retry queue (`notification_delivery_state.json`, `notify_queue.json`) | `notify_engine/alertnotifiers.py` + `NotificationServer` | `generation_file_lock` across gate updates, `.previous` repair, enqueue, and retry flush | cross-process serialized |
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
