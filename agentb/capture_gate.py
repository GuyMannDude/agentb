"""
Mnemo Cortex — capture pause gate (v4.1)
========================================
"I'm about to do something sensitive — stop recording for a bit."

The auto-capture pipeline syncs terminal activity every 60 seconds. During a
credential rotation or any secret-handling work, even redaction-pattern misses
are a risk — the only safe capture is no capture. The gate pauses ambient
capture (/ingest, and auto-capture-shaped /writeback) server-wide.

Forgetting to unpause would silently lobotomize the memory system, so the gate
is a dead-man switch, not a toggle: every pause carries an expiry (default 15
minutes, hard cap 4 hours) and the watchdog is lazy — any request that checks
the gate after expiry auto-resumes it. No background thread to crash, and the
state survives server restarts because it lives in a file.

Deliberate manual saves are NEVER gated — pausing ambient capture while saving
the *why* of the sensitive operation is exactly the intended workflow.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("agentb.capture_gate")

DEFAULT_PAUSE_MINUTES = 15
MAX_PAUSE_MINUTES = 240


class CaptureGate:
    """File-backed, server-wide pause switch with auto-resume on expiry."""

    def __init__(self, data_dir: Path):
        self.pause_file = Path(data_dir) / "capture-pause.json"

    def _read(self) -> Optional[dict]:
        if not self.pause_file.exists():
            return None
        try:
            return json.loads(self.pause_file.read_text())
        except Exception as e:
            # Unreadable state file: fail toward capturing (the system's normal
            # mode) but say so — a corrupt pause file must not be a silent
            # permanent pause.
            log.warning(f"capture-pause file unreadable ({e}) — treating as not paused")
            return None

    def pause(self, minutes: Optional[int] = None, reason: str = "") -> dict:
        mins = minutes if minutes and minutes > 0 else DEFAULT_PAUSE_MINUTES
        mins = min(mins, MAX_PAUSE_MINUTES)
        now = time.time()
        state = {
            "paused_at": now,
            "resume_at": now + mins * 60,
            "minutes": mins,
            "reason": reason or "(no reason given)",
        }
        self.pause_file.parent.mkdir(parents=True, exist_ok=True)
        self.pause_file.write_text(json.dumps(state, indent=2))
        log.warning(f"⏸ Capture PAUSED for {mins} min — {state['reason']} (auto-resumes)")
        return self.status()

    def resume(self) -> dict:
        was_paused = self.pause_file.exists()
        self.pause_file.unlink(missing_ok=True)
        if was_paused:
            log.warning("▶ Capture RESUMED (manual)")
        return self.status()

    def is_paused(self) -> bool:
        """Lazy watchdog: reading the gate after expiry auto-resumes it."""
        state = self._read()
        if state is None:
            return False
        if time.time() >= float(state.get("resume_at", 0)):
            self.pause_file.unlink(missing_ok=True)
            log.warning("▶ Capture RESUMED (pause expired — watchdog)")
            return False
        return True

    def status(self) -> dict:
        state = self._read()
        if state is None or time.time() >= float(state.get("resume_at", 0)):
            return {"paused": False}
        return {
            "paused": True,
            "reason": state.get("reason", ""),
            "paused_at": state.get("paused_at"),
            "resume_at": state.get("resume_at"),
            "remaining_seconds": int(float(state["resume_at"]) - time.time()),
        }
