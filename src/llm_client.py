from typing import Protocol

from src.config import Settings


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        compact_prompt = " ".join(prompt.split())
        excerpt = compact_prompt[:220]
        return (
            "Mock response: this deterministic response is generated for local tests and "
            f"offline demos. Prompt excerpt: {excerpt}"
        )


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.ollama_model)
    if settings.llm_provider in {"openai", "openai_compatible"}:
        if not settings.openai_base_url or not settings.openai_api_key or not settings.openai_model:
            return MockLLMClient()
        return OpenAICompatibleClient(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.openai_model,
        )
    return MockLLMClient()
