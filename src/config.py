from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: Path = Path("vectorstore")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("AI_RA_LLM_PROVIDER", cls.llm_provider).strip().lower(),
            ollama_base_url=os.getenv("AI_RA_OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            ollama_model=os.getenv("AI_RA_OLLAMA_MODEL", cls.ollama_model),
            openai_base_url=os.getenv("AI_RA_OPENAI_BASE_URL", ""),
            openai_api_key=os.getenv("AI_RA_OPENAI_API_KEY", ""),
            openai_model=os.getenv("AI_RA_OPENAI_MODEL", ""),
            embedding_model=os.getenv("AI_RA_EMBEDDING_MODEL", cls.embedding_model),
            chroma_path=Path(os.getenv("AI_RA_CHROMA_PATH", str(cls.chroma_path))),
        )
