# Model Card

## Embedding Model

Default semantic embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is lightweight enough for local demos and produces useful sentence embeddings for retrieval. The app also includes a deterministic hash embedding model for tests and quick smoke checks.

## LLM Providers

Supported providers:

- `mock`: deterministic local response for tests and offline setup.
- `ollama`: local generation through Ollama.
- `openai_compatible`: any API exposing an OpenAI-style chat completions endpoint.

The default provider is `mock`, so the app starts without keys or a local model server.

## Limitations

- Hash embeddings are for smoke tests, not high-quality semantic retrieval.
- Local Ollama answer quality depends on the selected model.
- API-based models may provide better summaries and comparisons but require credentials.
- The system cannot verify claims outside the indexed paper chunks.
