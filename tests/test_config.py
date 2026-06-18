from src.config import Settings


def test_settings_defaults_are_local_first(monkeypatch):
    for key in [
        "AI_RA_LLM_PROVIDER",
        "AI_RA_OLLAMA_BASE_URL",
        "AI_RA_OLLAMA_MODEL",
        "AI_RA_OPENAI_BASE_URL",
        "AI_RA_OPENAI_API_KEY",
        "AI_RA_OPENAI_MODEL",
        "AI_RA_EMBEDDING_MODEL",
        "AI_RA_CHROMA_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings.from_env()

    assert settings.llm_provider == "mock"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert str(settings.chroma_path) == "vectorstore"
