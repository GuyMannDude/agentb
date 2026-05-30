# tools/ — Facts seeder

A small toolkit for keeping Mnemo Cortex's Phase 3 **Facts** table seeded with
the canonical truths you already know.

## Why seed facts?

Semantic recall and the Facts table answer different questions. Recall is fuzzy
matching over memories; Facts are exact `(entity, attribute, value)` lookups.
When the Facts table is **empty**, a stale session-log memory wins a lookup it
should lose — an agent "remembers" an old model name or a retired port because
nothing canonical contradicted it.

Seeding the truths you already know, at `confidence=verified`, gives Mnemo a
hard anchor. A verified Fact outranks any contradicting fuzzy recall.

## Files

| File | Purpose |
|------|---------|
| `seed-facts.py` | Loader. Reads a YAML of claims, asserts each as a Fact. Idempotent. |
| `seed-facts.example.yaml` | Worked example. Copy to `seed-facts.yaml` and edit. |
| `seed-facts-post-commit.sh` | Git hook: re-seed when your YAML changes. |
| `seed-facts-nightly.sh` | Cron wrapper: nightly safety-net re-seed. |

## Quick start

```bash
# 1. Copy the example and fill in your truths.
cp tools/seed-facts.example.yaml tools/seed-facts.yaml
$EDITOR tools/seed-facts.yaml

# 2. Dry-run to preview (writes nothing).
.venv/bin/python tools/seed-facts.py --yaml tools/seed-facts.yaml --dry-run

# 3. Commit for real.
.venv/bin/python tools/seed-facts.py --yaml tools/seed-facts.yaml
```

Dependencies (`httpx`, `pyyaml`) ship with Mnemo Cortex — run it with the repo
`.venv` and there's nothing extra to install.

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `MNEMO_URL` | `http://127.0.0.1:50001` | Base URL of the service (`--mnemo-url` overrides). |
| `MNEMO_AUTH_TOKEN` | _(unset)_ | Sent as `X-API-KEY`. Only needed when the service enforces auth (non-loopback binds). Omit for the default loopback setup. |

## Keeping it in sync (optional)

Two ways to keep the table fresh as your truths change. Both are **opt-in** and
default-off in `robot.install`; the installer can wire them up for you, or set
them up by hand:

**Post-commit hook** — re-seeds the moment you commit an edit to your YAML:

```bash
ln -sf "$PWD/tools/seed-facts-post-commit.sh" \
       "$(git rev-parse --show-toplevel)/.git/hooks/post-commit"
```

**Nightly cron** — catches drift the hook can't see (edits committed from
another machine, a Mnemo restart):

```cron
10 3 * * * SEED_FACTS_REPO=$HOME/my-brain /path/to/mnemo-cortex/tools/seed-facts-nightly.sh
```

Each script documents its own env knobs in its header.

## Idempotency & conflicts

Re-running is safe: facts whose `(entity, attribute, value)` already match are
skipped (`MATCH`). A different value at the same attribute is reported as a
`CONTRADICTION` — in `--dry-run` nothing is written, so you can inspect the
conflict before deciding. The confidence ladder is `verified > high_probability
> false`; a lower-confidence assertion can't silently overwrite a higher one.
