from src.llm_client import MockLLMClient
from src.prompts import build_rag_prompt


def test_mock_llm_client_returns_deterministic_text():
    client = MockLLMClient()
    response = client.generate("Question: What is RAG?")

    assert "Mock response" in response


def test_rag_prompt_requires_evidence():
    prompt = build_rag_prompt("What is RAG?", ["[1] Context text"])

    assert "Only answer based on the provided context" in prompt
    assert "Do not invent sources" in prompt
    assert "[1] Context text" in prompt
