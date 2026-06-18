# AI Research Assistant MVP Design

## Goal

Build a portfolio-ready MVP for an AI research assistant that demonstrates a complete retrieval-augmented generation pipeline for academic papers. The first version should run locally by default, support optional OpenAI-compatible API models, and be strong enough to publish on GitHub as an application project.

## Product Scope

The MVP supports five workflows:

1. Search arXiv papers by keyword and save metadata.
2. Upload local PDF papers and extract text.
3. Chunk extracted text with source metadata.
4. Build and query a ChromaDB vector store with sentence-transformer embeddings.
5. Ask questions, summarize papers, and compare papers with evidence-based outputs.

Out of scope for the first implementation:

- Semantic Scholar integration.
- Zotero or BibTeX export.
- Citation graph visualization.
- Docker packaging.
- Multi-embedding benchmark UI.

These are future-work items after the core loop is reliable.

## Architecture

The application is split into a Streamlit UI layer and reusable pipeline modules.

```text
app/streamlit_app.py
    -> src/arxiv_search.py
    -> src/pdf_loader.py
    -> src/text_cleaning.py
    -> src/chunking.py
    -> src/embeddings.py
    -> src/vector_store.py
    -> src/llm_client.py
    -> src/summarizer.py
    -> src/paper_compare.py
    -> src/rag_pipeline.py
    -> src/evaluation.py
```

The UI should orchestrate workflows but not contain core RAG logic. Each `src/` module should expose small, testable functions or classes.

## Data Flow

```text
User keyword or uploaded PDF
-> paper metadata and/or extracted text
-> cleaned text
-> chunks with paper title, paper id, page number, chunk id, and source
-> embeddings
-> ChromaDB collection
-> top-k retrieval
-> LLM prompt with retrieved evidence
-> answer, evidence list, confidence
```

For arXiv search, the MVP records metadata first. PDF download can be supported when URLs are available, but local upload remains the reliable path for indexing.

## Model Strategy

Embeddings default to a lightweight local `sentence-transformers` model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The LLM layer uses a provider abstraction:

- `ollama`: default local provider.
- `openai_compatible`: optional provider configured by environment variables.
- `mock`: deterministic fallback used in tests and when no local LLM is available.

The application should not require an API key to start.

## Chunking Strategy

The MVP uses word-based chunking with overlap as a practical first version:

- Target size: about 700 words.
- Overlap: about 120 words.
- Metadata preserved on every chunk.

This is chosen because it is transparent, easy to test, and robust without requiring tokenizer-specific dependencies. The documentation should explain that token-aware chunking is a future improvement.

## RAG Answer Contract

Every RAG response follows this structure:

```text
Answer:
...

Evidence:
[1] Paper title, page/chunk, source
[2] Paper title, page/chunk, source

Confidence:
High / Medium / Low
```

The system must not present unsupported claims as facts. If retrieved context is empty or weak, the answer should say that the current paper library is insufficient.

## Streamlit Pages

The MVP includes these user-facing areas:

1. `Search Papers`: search arXiv, inspect metadata, save CSV.
2. `Upload PDFs`: upload PDFs, extract text, chunk, and index.
3. `Ask Questions`: query the vector store and display answer, citations, and retrieved chunks.
4. `Compare Papers`: create a structured comparison table from selected papers or provided summaries.
5. `Evaluation`: run a small manual evaluation set and show retrieval/answer checks.

The interface should feel like a practical research tool rather than a landing page.

## Evaluation

The MVP includes a lightweight evaluation module and report. It should support:

- A small set of demo questions.
- Recall-style retrieval checks for expected paper titles or keywords.
- Answer checks for citation presence and insufficient-evidence behavior.

The first version can include a template evaluation dataset if no real indexed papers are present.

## Tests

Initial tests should cover:

- Text chunking preserves metadata and overlap.
- PDF extraction returns page-aware text or clear errors.
- Vector retrieval returns ranked chunks.
- RAG response formatting includes evidence and confidence.

Tests should use small fixtures and mock providers where possible, so CI and local runs do not require an LLM server.

## Documentation

The GitHub-facing documentation should include:

- Project overview and motivation.
- Architecture diagram.
- Local setup instructions.
- Model/provider configuration.
- Example queries.
- Evaluation method and limitations.
- Future work.

Reports should live under `reports/`, and prompt details under `docs/prompt_design.md`.

## Success Criteria For MVP

The MVP is acceptable when:

- `streamlit run app/streamlit_app.py` starts the app.
- arXiv search returns a metadata table.
- PDF upload extracts and chunks text.
- chunks can be embedded and stored in ChromaDB.
- a question returns an answer with evidence and confidence.
- paper comparison produces a structured table.
- tests pass for core pipeline behavior.
- README explains how to run and what the project demonstrates.
