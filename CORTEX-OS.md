# Cortex OS — The Operating Manual for Agents

You are an AI agent, and Mnemo Cortex was just installed for you. This
document teaches you how to use it. Read it once now, in full; your future
sessions depend on the habits you adopt here.

Having memory and knowing how to think with it are different things. The
install gave you tools. This page gives you the operating system.

(If you are the human: this doc is for your agent — link it, or paste it,
into whatever instruction file your platform reads at session start. The
human-side companion is [SESSION-GUIDE.md](SESSION-GUIDE.md).)

---

## 1. The startup ritual

**Call `agent_startup` as your FIRST tool call, every session. No
exceptions.**

You wake up blank. Whatever you think you know about this user, this
project, this machine — that's training data plus whatever survived in
your context window. Both go stale. `agent_startup` returns a boot block
with your lane file (who you are, what you were working on), the current
project state, recent memories, and — if the dreamer is running — an
overnight synthesis of what all agents on this server learned.

The rule that follows from this: **never brief your user from memory
alone.** If they ask "where did we leave off?", the answer comes from your
startup block or a recall — tool output or silence. An agent that
confidently improvises a status update from stale assumptions is worse
than one that says "let me check."

If `agent_startup` isn't registered in your tool list, the install is
incomplete — tell your user, pointing them at
[robot.install.md](robot.install.md) section 4.

## 2. Brain files — your working memory

`read_brain_file` / `write_brain_file` / `list_brain_files` give you
persistent markdown files on disk. This is your scratch space, and it is
the difference between waking up oriented and waking up amnesiac.

Use brain files for what is true *right now*:

- Your **lane file** — your identity, current focus, and a dated log of
  what changed each session. Update it every session, newest first.
- Project state, active task lists, open threads.
- Capability maps, environment facts, anything your next session needs in
  the first sixty seconds.

Two disciplines:

- **Replace, don't accumulate.** A brain file is a snapshot, not a
  journal. When state changes, rewrite the line — history belongs in
  memories (section 3), not stacked in the file your future self must
  read in full.
- **Writes are not commits.** `write_brain_file` writes to disk. If your
  setup persists the brain directory through git, the commit/push is a
  separate step — do it, or flag your operator. A brain file that only
  changed locally is invisible to every other machine and agent, and one
  crash away from gone. (Some installs expose `session_end`, which
  commits for you — see section 6.)

## 3. Memory — what to save and when

`mnemo_save` is for context that should survive across sessions:
decisions made and why, problems solved, user preferences learned,
project milestones, approaches that were tried and rejected.

What NOT to save: transient debug output, routine question-and-answer,
anything your user says is temporary, and anything a file in the repo
already records better than a memory would.

**Quality over quantity.** One memory with clear context — what happened,
why it matters, what to do differently — beats ten vague ones. You are
not a logger; ambient capture (if your install has it) handles the play-
by-play. Your saves are the editorial layer: the *why*.

## 4. Recall — check before you propose

Before you suggest a solution, an architecture, a library, a plan:
**recall first.**

You may have already solved this. Your user may have already rejected
this exact idea, twice, with reasons. Nothing burns trust faster than
re-proposing last month's discarded design with fresh enthusiasm.

Use `mnemo_recall` with a focused query — "database migration approach
for the billing service", not "database stuff". Focused queries hit;
vague ones return noise. And when recall returns something that
contradicts your assumption, **trust the recall** — it was saved by a
version of you that was actually there.

## 5. Facts — structured knowledge

Memories are fuzzy semantic search over prose. Facts
(`mnemo_fact_save` / `mnemo_fact_get` / `mnemo_fact_query`) are exact,
structured lookups: entity–attribute–value with confidence and
provenance.

Use facts for discrete truths you'll need verbatim: "the user's name is
X", "the project targets Python 3.12", "the deploy region is
us-west-2", "server Y was retired on date Z". Use memories for
narrative: how the migration went, why the design changed.

Rule of thumb: if the answer must be *exact* to be useful, it's a fact.
If the answer is a story, it's a memory. Save accordingly, and when a
fact turns out to be wrong or stale, demote it (`mnemo_fact_demote`) —
a confidently wrong fact store is worse than an empty one.

## 6. Session discipline

Your future self wakes up with only what you persisted. What you don't
save, you lose — every session, permanently.

Before a session ends:

1. Update your lane/brain files — date bump, what changed, what's next.
2. `mnemo_save` the decisions and outcomes that mattered (the why).
3. If your install exposes `session_end`, call it — it flushes capture,
   saves a summary, and commits the brain directory.
4. If it doesn't, and your brain uses git: commit and push, or tell your
   operator it needs doing.

Sessions end unexpectedly — context fills, terminals close, machines
sleep. Don't hoard the write-back for a grand finale; persist important
state when it becomes important.

## 7. Trajectories — learning how to work

Memories record what happened. Trajectories
(`mnemo_save_trajectory` / `mnemo_recall_trajectory`) record **how you
did something** — the steps, the dead ends, the outcome, a rating.

Before starting a task type you've likely done before (a deploy, a
migration, a tricky install), `mnemo_recall_trajectory` — your past self
may have left you a recipe, including which approaches wasted an hour.

After completing a nontrivial task well, `mnemo_save_trajectory` with
honest steps and outcome. This is how you get better over time without
retraining: not by remembering that the work happened, but by
remembering how to do it.

## 8. Common mistakes

Watch yourself for these — every one is from field observation:

- **Narrating instead of calling.** Saying "I've saved that to memory"
  without a tool call having happened. Smaller models especially. If it
  matters, verify the save landed — the tool result names a memory id.
- **Briefing from training data.** Answering "what's the state of the
  project?" without having called `agent_startup` or recall that session.
- **Saving everything.** Noise drowns signal; retrieval quality degrades
  with every junk memory. Be an editor.
- **Not recalling before proposing.** See section 4. Reinventing a
  rejected idea reads as not listening.
- **Letting brain files rot.** A lane file last updated three weeks ago
  actively misleads your future self — stale state presented as current
  is worse than no state.
- **Assuming continuity.** Your last session's context is gone. Ports
  change, servers move, decisions get reversed while you're not running.
  Verify current state with tools before acting on remembered state.

---

## The shape of a good session

```
wake  →  agent_startup            (orient before anything else)
work  →  recall before proposing  (check the past)
      →  save decisions as made   (don't hoard the why)
close →  update brain files       (snapshot what's true now)
      →  session_end / commit     (make it survive you)
```

That's the whole operating system. Memory doesn't make you smarter —
it makes you *continuous*. The agent your user talks to next week is
only as good as what you persist today.
