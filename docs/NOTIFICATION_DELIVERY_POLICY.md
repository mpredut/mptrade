# Shared notification delivery policy

Updated: 2026-09-08. Policy owner: `alertnotifiers.py`, shared by provider bots and
the Python watchdogs through `verify_tools/watchdog_common.py`.

| Delivery | Local daily volume policy | Duplicate suppression |
| --- | --- | --- |
| Routine ntfy | Preserve `max(0, NTFY_DAILY_BUDGET - NTFY_URGENT_RESERVE)` | Yes |
| Urgent ntfy | No local daily cap, including after the old total is exhausted | Yes |
| Email, routine or urgent | No local daily cap | Yes |

The ntfy counter still counts all reserved attempts conservatively. The retained
reserve setting limits routine consumption to leave headroom at the provider;
it is no longer an upper bound on urgent attempts. Email's counter is telemetry
only. Legacy `EMAIL_DAILY_BUDGET` and `EMAIL_URGENT_RESERVE` overrides no longer
limit email. Existing state and counters do not need to be reset or deleted.

Urgency uses the shared title markers and watchdog source classification. The
`notify()` wrapper uses the same classification for automatic email. Routine
events do not automatically acquire an email copy; their existing opt-in remains.
Explicit email calls, including watchdog email, are not volume-capped locally.

## Provider limits and fallback

This policy cannot override ntfy or SMTP provider quotas, authentication failures,
or outages. An actual ntfy daily-quota response still blocks further ntfy requests
until the UTC daily state resets. Each urgent incident refused by that quota is
also passed to email with its original detail, including subsequent incidents
after the ntfy channel has been marked blocked. A generic quota warning is not
a substitute for forwarding the incident itself.

The phone method still returns failure when ntfy rejects it, even if the email
fallback succeeds. An email transport success means SMTP acceptance, not proof
of inbox delivery. Other phone errors keep their existing bounded retry behavior;
the shared `notify()` urgent path also attempts email independently.

Repeated identical events retain cross-process deduplication and the existing
cooldowns, so removing daily caps does not deliberately resend the same incident
on every poll. Attempt reservation precedes network I/O; this is not a durable
guaranteed-delivery queue. Existing explicit notification-disable switches remain.

## Regression coverage

`tests/test_notification_delivery_policy.py` covers exhausted legacy counters,
zero routine allowance, unlimited email with old overrides, urgent deduplication,
provider-quota fallback carrying each original incident, failed fallback reporting,
and consistent watchdog urgency. Tests mock HTTP/SMTP; they do not send live alerts.

## Retry expiry incidents

`order_retry_worker.py` writes a separate `retry_giveup` execution-audit event for
each expired intent. Notifications use the reusable `notification_digest.py`:

- The first batch for an incident is reported immediately.
- Further expiries are summarized every `RETRY_GIVEUP_SUMMARY_SEC` (900 seconds in
  `config.env`). The interval controls notifications only.
- Incidents are grouped by provider, symbol, side, kind, last refusal and expiry
  reason. A different incident is not suppressed by another incident's cooldown.
- Each summary contains the new count, total requested quantity and one sample
  intent ID. Quantity is not executed volume or account exposure.
- Pending counts and delivery-attempt reservations survive worker restart in
  `cachedb/order_retry_giveup_alerts.json`, beside the configured retry queue.
  Empty worker passes flush the final summary; quiet groups are then pruned.
- The audit remains best-effort. Digest state is locked and atomically replaced;
  invalid state is not silently reset. A digest error is logged and does not
  abort order processing or fall back to a per-intent notification storm.

This is aggregation, not a daily cap and not guaranteed delivery. Reservations
precede notification transport; a crash or transport failure can lose a delivery
attempt. Quarantine, unknown submissions, hard-stop incidents and unrelated
urgent alerts retain their existing paths. No retry TTL, attempt limit, price
gate, order ownership or missing-weight protection changes are included.

## Anomaly counting

`weight_policy_unavailable` means execution was blocked before submission. It is
classified as `execution_blocked`, not `blind`, with its own mandatory threshold
`ANOMALY_THRESH_EXECUTION_BLOCKED=25`. The existing anomaly cooldown still applies.
This remains worth reporting when persistent; it is not evidence of blind trading.

The structured `order_outcomes_YYYY-MM-DD.log` stream is authoritative for these
refusals when readable and baselined. Worker text provides fallback at startup,
daily rollover or read failure. Exact stdout/daily mirror lines are counted with
their maximum per-file multiplicity, preserving repeated events within a file
and identical messages from unrelated bots. Watchdog self-reports are excluded
so quoted examples cannot trigger new alarms. Other missing-data, authentication,
rate-limit and traceback categories remain visible.

Tests in `test_notification_digest.py`, `test_order_retry_worker.py` and
`test_anomaly_event_counting.py` exercise restart, concurrent reservations, final
empty-queue summaries, per-intent audit, corrupted state, attempt-limit wording,
59 refusals mirrored into three logs, daily rollover and existing cooldowns.

## Rtrade retry ownership audit

The versioned `config.env` enables the pair coordinator. Its limit-order
adapter explicitly sets `caller_owns_retry=True`; the common tracked lifecycle
persists and reconciles the pair's intent without the global retry worker. Its
hard-stop path also uses the pair-owned audited executor. These decisions must
be distinguished from the dormant legacy branch's repetitive BUY/SELL loops.

The legacy branch does not opt out consistently: ordinary quotes and follow-up
calls can enter the global outbox while its own loop retries. Reducing global TTL
does not establish a single owner and would affect unrelated strategies. A future
legacy migration must preserve persist-before-submit and ambiguous-response
recovery before opting out, or prevent new legacy attempts while the global
outbox owns the original intent. Do not simply add the opt-out flag everywhere,
hide all rtrade alerts, or discard accepted/ambiguous trackers.

The current patch does not change rtrade order policy. Local configuration and
call paths alone cannot establish which producer created a historical production
intent; attribution requires live configuration and retained per-intent evidence.
