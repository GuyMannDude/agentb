# E3 — do the HOT / L1 / L2 tiers serve anything?

**Date:** 2026-09-01/02 · **Server:** Mnemo Cortex 4.15.0 (live, post-Experiment-One ranker) · **Tenants:** all four configured.

Read-only `/context` probes (`run.sh` → the Experiment One `probe.py`, unchanged):
focus mode, 10 results, `session_log` excluded, 60 queries per tenant from the
same realistic list Experiment One used (187 on the first tenant). Logged per
hit: served position, tier, raw relevance. Only server side effect: the
access-count bump `/context` always performs on served memories.

| tenant | queries | HOT | L1 | VEC | L2 | L3 | p50 latency |
|---|---|---|---|---|---|---|---|
| A (Experiment One run) | 186 | 0 | 6 | 1849 | 0 | 0 | — |
| B | 60 | 0 | 0 | 600 | 0 | 0 | 280 ms |
| C | 60 | 0 | 0 | 600 | 0 | 0 | 273 ms |
| D | 60 | 0 | 0 | 600 | 0 | 0 | 273 ms |

**3,665 served chunks: VEC 99.8 %, L1 0.2 % (tenant A only), HOT / L2 / L3 zero.**
These are SERVED counts (post-trim), so they bound what the tiers pooled from
below, not above — a tier chunk that padded the pool and lost the re-rank is
invisible here. That is why the L3 gate was made uniform in the same change
(see CHANGELOG) rather than argued away with this table.

## The hole in the absence check, and its presence check

HOT rows are `session_log`, which these probes exclude — HOT = 0 above is by
construction. Re-probed with `exclude_categories: []` and single common words
on two tenants: still HOT 0. `/sessions` explained it: three of the four
tenants have ZERO hot or warm sessions; the fourth's hot sessions are a
scheduler's heartbeat polls. Presence probe on that tenant, query "heartbeat",
opted in → `HOT: 3` at the fixed 0.75 relevance. The tier works; heartbeat
noise is all it has to serve.
