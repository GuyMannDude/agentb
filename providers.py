"""
AgentB Provider Abstraction Layer
Pluggable backends for reasoning (LLM) and embedding models.
"""

from abc import ABC, abstractmethod
from typing import Optional
from agentb.config import ProviderConfig


# ─────────────────────────────────────────────
#  Base Classes
# ─────────────────────────────────────────────

class ReasoningProvider(ABC):
    """Base class for LLM reasoning providers (preflight checks, summarization)."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        """Generate a text completion."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
        pass


class EmbeddingProvider(ABC):
    """Base class for embedding providers (semantic search)."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
        pass


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def create_reasoning_provider(config: ProviderConfig) -> ReasoningProvider:
    """Create a reasoning provider from config."""
    providers = {
        "ollama": OllamaReasoning,
        "openai": OpenAIReasoning,
        "anthropic": AnthropicReasoning,
        "openrouter": OpenRouterReasoning,
        "google": GoogleReasoning,
    }
    cls = providers.get(config.provider)
    if not cls:
        raise ValueError(f"Unknown reasoning provider: {config.provider}. Options: {list(providers.keys())}")
    return cls(config)


def create_embedding_provider(config: ProviderConfig) -> EmbeddingProvider:
    """Create an embedding provider from config."""
    providers = {
        "ollama": OllamaEmbedding,
        "openai": OpenAIEmbedding,
        "huggingface": HuggingFaceEmbedding,
        "google": GoogleEmbedding,
        "openrouter": OpenRouterEmbedding,
    }
    cls = providers.get(config.provider)
    if not cls:
        raise ValueError(f"Unknown embedding provider: {config.provider}. Options: {list(providers.keys())}")
    return cls(config)


# ─────────────────────────────────────────────
#  Ollama Providers (local, free)
# ─────────────────────────────────────────────

class OllamaReasoning(ReasoningProvider):
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        import httpx
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(f"{self.config.api_base}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def health_check(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.api_base}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


class OllamaEmbedding(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        import httpx
        payload = {"model": self.config.model, "input": text}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{self.config.api_base}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [[]])
            return embeddings[0] if embeddings else []

    async def health_check(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.api_base}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────
#  OpenAI Providers
# ─────────────────────────────────────────────

class OpenAIReasoning(ReasoningProvider):
    def _base_url(self):
        return self.config.api_base or "https://api.openai.com/v1"

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(f"{self._base_url()}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url()}/models", headers=headers)
                return resp.status_code == 200
        except Exception:
            return False


class OpenAIEmbedding(EmbeddingProvider):
    def _base_url(self):
        return self.config.api_base or "https://api.openai.com/v1"

    async def embed(self, text: str) -> list[float]:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.model, "input": text}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{self._base_url()}/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def health_check(self) -> bool:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url()}/models", headers=headers)
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────
#  Anthropic Provider
# ─────────────────────────────────────────────

class AnthropicReasoning(ReasoningProvider):
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        import httpx
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        base = self.config.api_base or "https://api.anthropic.com/v1"
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(f"{base}/messages", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"] if data.get("content") else ""

    async def health_check(self) -> bool:
        return bool(self.config.api_key)


# ─────────────────────────────────────────────
#  OpenRouter Provider (unified gateway)
# ─────────────────────────────────────────────

class OpenRouterReasoning(ReasoningProvider):
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/GuyMannDude/agentb",
            "X-Title": "AgentB",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        base = self.config.api_base or "https://openrouter.ai/api/v1"
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        return bool(self.config.api_key)


class OpenRouterEmbedding(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.model, "input": text}
        base = self.config.api_base or "https://openrouter.ai/api/v1"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{base}/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def health_check(self) -> bool:
        return bool(self.config.api_key)


# ─────────────────────────────────────────────
#  Google/Gemini Provider
# ─────────────────────────────────────────────

class GoogleReasoning(ReasoningProvider):
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        import httpx
        base = self.config.api_base or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.config.model}:generateContent?key={self.config.api_key}"

        contents = [{"parts": [{"text": prompt}]}]
        payload = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return parts[0].get("text", "") if parts else ""
            return ""

    async def health_check(self) -> bool:
        return bool(self.config.api_key)


class GoogleEmbedding(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        import httpx
        base = self.config.api_base or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.config.model}:embedContent?key={self.config.api_key}"
        payload = {"content": {"parts": [{"text": text}]}}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("embedding", {}).get("values", [])

    async def health_check(self) -> bool:
        return bool(self.config.api_key)


# ─────────────────────────────────────────────
#  HuggingFace Embedding (local or API)
# ─────────────────────────────────────────────

class HuggingFaceEmbedding(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        import httpx
        if self.config.api_base:
            # Local inference server (text-embeddings-inference)
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.config.api_base}/embed",
                    json={"inputs": text}
                )
                resp.raise_for_status()
                return resp.json()[0]
        else:
            # HuggingFace Inference API
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.config.model}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json={"inputs": text}, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data[0], list):
                    # Mean pooling
                    import numpy as np
                    return np.mean(data, axis=0).tolist() if len(data) > 1 else data[0]
                return data

    async def health_check(self) -> bool:
        return True
