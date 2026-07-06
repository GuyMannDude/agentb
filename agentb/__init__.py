"""Mnemo Cortex Recall — Exact-match memory search via SQLite FTS5.

Complements Mnemo's semantic/vector search with precise keyword and
entity-based recall. Originally conceived as 'claw-recall' by AL.

Two search modes, one memory system:
    - Mnemo /context  → "What do you remember about Easter?" (semantic, fuzzy)
    - Mnemo recall    → "What was the Shopify API key?" (exact, precise)
"""
from pathlib import Path


def _resolve_version() -> str:
    """Single source of truth for the release version.

    The live deployment runs straight from a git checkout, where an installed
    dist's metadata can be stale (a `pip install` from months ago shadows every
    `git pull` since). So: a pyproject.toml sitting next to the package wins;
    dist metadata is only trusted when there is no checkout (a real wheel
    install, e.g. from PyPI).
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        import tomllib
        with pyproject.open("rb") as f:
            version = tomllib.load(f).get("project", {}).get("version")
        if version:
            return version
    import importlib.metadata
    try:
        return importlib.metadata.version("mnemo-cortex")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()
