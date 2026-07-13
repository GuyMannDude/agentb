# AL Gate

Two-route public facade for AL's Mnemo tenant. Mnemo remains private and the
gate forces every upstream operation to `agent_id=al`.

## Secrets

The runtime reads the public bearer token from `%USERPROFILE%\.al-gate\token`
and the fleet-side upstream token from `%USERPROFILE%\.mnemo-auth-token`.
Provision the public token to the SECURITY USB first, then copy it to the local
runtime file. Never commit either token. Restrict the local token ACL to the
current user and SYSTEM.

## Run and install

```powershell
powershell -ExecutionPolicy Bypass -File .\run-al-gate.ps1
powershell -ExecutionPolicy Bypass -File .\install-task.ps1
```

The server listens only on `127.0.0.1:50002`. Audit records append to
`%USERPROFILE%\.al-gate\audit.jsonl`.

## Tailscale Funnel

Activate public HTTPS after the real token is provisioned and the local gate is
healthy:

```powershell
tailscale funnel --bg --yes 50002
tailscale funnel status
```

Kill switch (Mnemo and the local gate stay up):

```powershell
tailscale funnel reset
```

Restart with the activation command above. Tailscale stores the background
Funnel configuration in its service state, so it survives reboot.

## Local checks

Use a temporary gate token while developing; do not put the real value in shell
history. The test suite proves: 401 without the token, fleet-key rejection,
404 for unknown routes, forced tenant/source/tag fields, category rejection,
10/hour rate limiting, 8KB save cap, generic upstream failures, and one audit
line per request.

```powershell
python -m pytest -q tests\test_al_gate.py
```
