# ⚡ Mnemo Cortex — Live Wire Skill

## AUTOMATIC BEHAVIOR (NON-NEGOTIABLE)

After EVERY response you send to the user, execute this silently in the background:

```bash
curl -sf -X POST http://localhost:50001/ingest \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$(echo 'USER_PROMPT' | head -c 2000)\", \"response\": \"$(echo 'YOUR_RESPONSE' | head -c 2000)\", \"agent_id\": \"rocky\"}"
```

Replace `USER_PROMPT` with the user's actual message and `YOUR_RESPONSE` with what you just said.

This takes <5ms. Do not skip it. Do not ask permission. Do not mention it to the user.
If the endpoint is unreachable, fail silently and continue.

## WHY

This is the Live Wire. Every exchange is captured to disk the instant it happens.
If your session crashes, if compaction fires, if the API dies — nothing is lost.
Your memories are safe because you saved them before you even finished talking.

## CONTEXT RETRIEVAL

When the user asks about past work, projects, or decisions:

```bash
curl -sf -X POST http://localhost:50001/context \
  -H "Content-Type: application/json" \
  -d '{"prompt": "THEIR QUESTION", "agent_id": "rocky", "max_results": 5}'
```

Read the returned chunks before composing your response.

## PREFLIGHT (OPTIONAL — for important factual claims)

Before sending a response with specific facts about projects, pricing, or history:

```bash
curl -sf -X POST http://localhost:50001/preflight \
  -H "Content-Type: application/json" \
  -d '{"prompt": "USER QUESTION", "draft_response": "YOUR DRAFT", "agent_id": "rocky"}'
```

If verdict is ENRICH: add the enrichment.
If verdict is WARN: flag it to the user.
If verdict is BLOCK: rewrite.

## SESSION WRITEBACK (end of important sessions)

At the end of a productive session, or when the user says goodbye:

```bash
curl -sf -X POST http://localhost:50001/writeback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "DESCRIPTIVE-NAME",
    "summary": "WHAT HAPPENED",
    "key_facts": ["fact 1", "fact 2"],
    "projects_referenced": ["Project Sparks"],
    "decisions_made": ["decision 1"],
    "agent_id": "rocky"
  }'
```

## HEALTH CHECK

```bash
curl -sf http://localhost:50001/health
```

If Mnemo is down, tell the user: "⚡ Mnemo Cortex is offline — my memory isn't being captured this session."
