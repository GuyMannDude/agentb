"""Shared filesystem helpers."""
import os
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """tmp + os.replace so a reader can never see a half-written file and a
    crash mid-write can never destroy the previous contents.

    Always UTF-8: the platform default is cp1252 on Windows, which dies on
    the first '→' in a trajectory line (and JSON/JSONL are UTF-8 by spec)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Bytes twin of atomic_write_text — same tmp + os.replace guarantee."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
