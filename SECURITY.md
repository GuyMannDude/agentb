# Security Policy

## Supported versions

Only the **latest release** receives security fixes. Mnemo Cortex moves fast (see
[CHANGELOG.md](CHANGELOG.md)) — if you're on an older version, upgrade first and re-test.

## Reporting a vulnerability

Please **do not open a public issue for security problems.**

- Preferred: use GitHub's private vulnerability reporting —
  [Report a vulnerability](https://github.com/GuyMannDude/mnemo-cortex/security/advisories/new)
- You'll get a human response. This is a one-maker project, so give it a few days;
  confirmed issues get fixed with priority over everything else.

## Scope notes

- Mnemo Cortex is designed to run **local-first** — bound to localhost or a private
  (e.g. Tailscale) network. Deployments that expose the server to the public internet
  are outside the intended threat model, but auth-bypass or data-leak issues are always
  in scope regardless of deployment.
- The Cortex Stick supports at-rest encryption (AES-256-SIV, key never on the stick).
  Weaknesses in that scheme are very much in scope.
- Prompt-injection resistance of memory content is an active area — reports with
  reproductions are welcome.
