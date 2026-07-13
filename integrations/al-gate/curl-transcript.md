# AL Gate local curl transcript

Run 2026-07-12 PT on IGOR-2 against `127.0.0.1:50002` with an ephemeral
development bearer token. The bearer value is omitted. Mnemo remained on
`127.0.0.1:50001` with its fleet token known only to the gate process.

```text
POST /recall, no Authorization header
{"detail":"Unauthorized"}
HTTP 401

POST /facts, valid gate bearer
{"detail":"Not Found"}
HTTP 404

POST /recall, valid gate bearer
request: {"prompt":"where did the name Project Sparks come from",
          "agent_id":"cc","max_results":1}
response: {"chunks":[...],"total_found":1,...,"agent_id":"al",...}
HTTP 200

Eleventh authorized request inside one hour
{"detail":"Rate limit exceeded"}
HTTP 429

POST /save with a 9KB body
{"detail":"Save request exceeds 8KB"}
HTTP 413
```

The forced-tenant recall returned the expected AL genesis memory and the audit
line recorded only the request summary:

```json
{"op":"/recall","size":90,"snippet":"where did the name Project Sparks come from","status":200}
```

Automated tests additionally verify that the fleet token receives 401 on the
public side, save requests force `agent_id=al` and `source=user`, the
`al-bridge` tag is deduplicated, disallowed categories never reach upstream,
upstream errors are generic, and every request receives an audit record.
