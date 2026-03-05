"""
AgentB Configuration
Loads agentb.yaml and provides typed access to all settings.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_CONFIG_PATHS = [
    Path("agentb.yaml"),
    Path("agentb.yml"),
    Path.home() / ".config" / "agentb" / "agentb.yaml",
    Path("/etc/agentb/agentb.yaml"),
]


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    provider: str = "ollama"          # ollama | openai | anthropic | openrouter | google | huggingface
    model: str = ""
    api_key: str = ""                 # or env var reference like ${OPENAI_API_KEY}
    api_base: str = ""                # custom endpoint URL
    timeout: float = 30.0
    extra: dict = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Configuration for the storage backend."""
    backend: str = "json"             # json | sqlite | postgres
    path: str = ""                    # for json/sqlite
    connection_string: str = ""       # for postgres
    extra: dict = field(default_factory=dict)


@dataclass
class CacheConfig:
    """L1/L2/L3 cache settings."""
    l1_max_bundles: int = 50
    l1_ttl_seconds: int = 86400       # 24 hours
    l1_similarity_threshold: float = 0.75
    l2_similarity_threshold: float = 0.5
    l3_similarity_threshold: float = 0.4


@dataclass
class ServerConfig:
    """Server settings."""
    host: str = "0.0.0.0"
    port: int = 50001
    cors_origins: list = field(default_factory=lambda: ["*"])
    auth_token: str = ""              # optional API key for securing endpoints


@dataclass
class AgentBConfig:
    """Root configuration object."""
    reasoning: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        provider="ollama", model="qwen2.5:32b-instruct"
    ))
    embedding: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        provider="ollama", model="nomic-embed-text"
    ))
    storage: StorageConfig = field(default_factory=StorageConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    data_dir: str = ""
    log_level: str = "info"
    agents: dict = field(default_factory=dict)  # multi-agent isolation


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} references in config values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key, "")
    return value


def _build_provider(data: dict) -> ProviderConfig:
    """Build a ProviderConfig from a dict."""
    return ProviderConfig(
        provider=data.get("provider", "ollama"),
        model=data.get("model", ""),
        api_key=_resolve_env(data.get("api_key", "")),
        api_base=_resolve_env(data.get("api_base", "")),
        timeout=data.get("timeout", 30.0),
        extra=data.get("extra", {}),
    )


def _build_storage(data: dict) -> StorageConfig:
    """Build a StorageConfig from a dict."""
    return StorageConfig(
        backend=data.get("backend", "json"),
        path=_resolve_env(data.get("path", "")),
        connection_string=_resolve_env(data.get("connection_string", "")),
        extra=data.get("extra", {}),
    )


def load_config(path: Optional[str] = None) -> AgentBConfig:
    """
    Load configuration from YAML file.
    Search order: explicit path → agentb.yaml → ~/.config/agentb/ → /etc/agentb/
    Falls back to defaults if no config found.
    """
    config_path = None

    if path:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        # Also check AGENTB_CONFIG env var
        env_path = os.environ.get("AGENTB_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            for candidate in DEFAULT_CONFIG_PATHS:
                if candidate.exists():
                    config_path = candidate
                    break

    if not config_path or not config_path.exists():
        # Return defaults
        return _apply_defaults(AgentBConfig())

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return _parse_config(raw)


def _parse_config(raw: dict) -> AgentBConfig:
    """Parse raw YAML dict into AgentBConfig."""
    cfg = AgentBConfig()

    if "reasoning" in raw:
        cfg.reasoning = _build_provider(raw["reasoning"])
    if "embedding" in raw:
        cfg.embedding = _build_provider(raw["embedding"])
    if "storage" in raw:
        cfg.storage = _build_storage(raw["storage"])
    if "cache" in raw:
        c = raw["cache"]
        cfg.cache = CacheConfig(
            l1_max_bundles=c.get("l1_max_bundles", 50),
            l1_ttl_seconds=c.get("l1_ttl_seconds", 86400),
            l1_similarity_threshold=c.get("l1_similarity_threshold", 0.75),
            l2_similarity_threshold=c.get("l2_similarity_threshold", 0.5),
            l3_similarity_threshold=c.get("l3_similarity_threshold", 0.4),
        )
    if "server" in raw:
        s = raw["server"]
        cfg.server = ServerConfig(
            host=s.get("host", "0.0.0.0"),
            port=s.get("port", 50001),
            cors_origins=s.get("cors_origins", ["*"]),
            auth_token=_resolve_env(s.get("auth_token", "")),
        )
    if "data_dir" in raw:
        cfg.data_dir = _resolve_env(raw["data_dir"])
    if "log_level" in raw:
        cfg.log_level = raw["log_level"]
    if "agents" in raw:
        cfg.agents = raw["agents"]

    return _apply_defaults(cfg)


def _apply_defaults(cfg: AgentBConfig) -> AgentBConfig:
    """Fill in smart defaults based on what's configured."""
    if not cfg.data_dir:
        cfg.data_dir = str(Path.home() / ".agentb")
    if not cfg.storage.path:
        cfg.storage.path = cfg.data_dir
    if not cfg.reasoning.model:
        cfg.reasoning.model = "qwen2.5:32b-instruct" if cfg.reasoning.provider == "ollama" else "gpt-4o-mini"
    if not cfg.embedding.model:
        cfg.embedding.model = "nomic-embed-text" if cfg.embedding.provider == "ollama" else "text-embedding-3-small"
    if not cfg.reasoning.api_base and cfg.reasoning.provider == "ollama":
        cfg.reasoning.api_base = "http://localhost:11434"
    if not cfg.embedding.api_base and cfg.embedding.provider == "ollama":
        cfg.embedding.api_base = "http://localhost:11434"
    return cfg
