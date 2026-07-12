# Cortex Stick — sneakernet for AI memory

> Your AI doesn't know you. This does. And it fits in your pocket.

You work from two desks — home and the shop, home and the office. Both
machines run Mnemo, and they drift: the decision you saved at one desk
doesn't exist at the other. The usual fixes put your AI's working memory
on somebody else's wire (cloud sync) or need infrastructure you don't
want to run (VPN, tailnet).

The Cortex Stick is a USB stick that works as a **courier** between two
full Mnemo installations. It is not a server — there's no database engine,
no embedder, nothing running on it. It carries the delta:

- **memories** — the per-agent memory JSONs (the truth files)
- **trajectories** — the append-only "how we did it" recipe logs
- **brain** — optionally, your brain/notes git repo (the stick holds a
  bare repo both machines push/pull through)
- **pad** — a free-form folder for dragging in-flight work between desks

Plug it in, sync, pull it out, carry it, plug it in. The other machine
catches up. No cloud, no VPN, no account.

## Quick start

```bash
# once, on any machine, with the stick mounted:
mnemo-cortex stick init /media/you/USB

# then, at each desk, whenever the stick is in:
mnemo-cortex stick sync

# or let it happen automatically while you work:
mnemo-cortex stick watch      # syncs on plug-in, re-syncs while present
```

`stick status` shows what would travel in each direction without changing
anything, plus when each machine last synced.

## How it stays safe

- **Only truth files cross.** Vector indexes, caches, and sidecar files are
  derived data — each machine rebuilds its own. New memories are recallable
  immediately via disk truth; embeddings catch up on the server's next
  backfill pass.
- **Every sync is a 3-way merge.** The stick remembers what each machine
  had at last sync, so it can tell "new on the other side" from "deleted
  here" — no silent overwrites, no resurrection of deleted memories.
- **Conflicts never destroy data.** If the same memory was edited at both
  desks, one version wins deterministically and the loser is preserved on
  the stick under `state/conflicts/`. If one desk edited what the other
  deleted, the edit wins. Trajectory logs union-merge — append-only truth
  never loses a row.
- **Nothing moves until every check passes.** A sync first *plans* the whole
  run — every guard fires before a single byte is copied — so a refusal
  really does mean nothing was changed, on either machine or the stick.
- **Yank-proof commits.** Files are hashed into a manifest that is written
  last; a stick yanked mid-sync fails verification on the next plug-in and
  the sync refuses, loudly, instead of merging from a torn state. "Safe to
  remove ✓" prints only after every written file is readback-verified. If a
  yank does tear a generation, `mnemo-cortex stick repair` rebuilds the
  manifest from the stick's contents and the next sync merges from there —
  no manual surgery, nothing deleted by the repair.
- **Massacre guard.** If a sync would delete more than a quarter of a
  store (a wiped or replaced machine), it refuses and explains, and only
  proceeds with `--force`.

## What v1 does not do

- **No encryption.** The stick is plaintext by design in v1 — contents are
  auditable by eye, and that cuts both ways: anyone holding the stick can
  read everything on it. Treat it like a notebook full of your working
  memory. (Filesystem-level encryption — BitLocker To Go, VeraCrypt, LUKS —
  works fine underneath; the stick doesn't care what the volume is made of.)
- **No session logs.** Raw session capture stays on each machine; the
  dreamer digests locally.
- **No facts table sync** (`facts.sqlite`) yet.
- **Two-desk, one-human.** Multi-user shared lanes are designed for
  (the merge machinery is already order-safe) but not shipped.

## Files on the stick

```
<mount>/cortex/
  passport.json        which stick this is, which machines it has met
  manifest.json        per-file SHA-256 of the current generation
  memories/<agent>/    memory JSONs + trajectory JSONLs, per agent
  brain/brain.git      bare git repo (if you courier a brain)
  pad/                 yours — drag anything
  state/               per-machine inventories, conflict archive, lock
```

Everything is a plain file. If you ever stop using the tool, your data is
sitting right there, readable.
