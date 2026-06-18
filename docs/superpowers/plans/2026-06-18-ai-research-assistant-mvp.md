# AI Research Assistant MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Streamlit MVP for literature search, PDF ingestion, chunking, embeddings, Chroma-backed retrieval, evidence-based RAG answers, paper summaries, comparison, and lightweight evaluation.

**Architecture:** Keep the Streamlit app thin and put reusable behavior in focused `src/` modules. The pipeline is local-first: sentence-transformers and ChromaDB for indexing, Ollama by default for generation, OpenAI-compatible APIs as an optional provider, and deterministic mock behavior for tests.

**Tech Stack:** Python 3.10+, Streamlit, pandas, arxiv, PyMuPDF, sentence-transformers, ChromaDB, requests, pytest.

---

## File Structure

- `pyproject.toml`: package metadata, dependencies, pytest config, ruff config.
- `.env.example`: documented provider and model settings.
- `app/streamlit_app.py`: Streamlit UI for search, upload, ask, compare, and evaluation pages.
- `src/config.py`: typed settings loaded from environment variables.
- `src/models.py`: dataclasses for papers, pages, chunks, retrieval results, and RAG answers.
- `src/arxiv_search.py`: arXiv search and metadata export.
- `src/pdf_loader.py`: page-aware PDF extraction using PyMuPDF.
- `src/text_cleaning.py`: deterministic whitespace and artifact cleanup.
- `src/chunking.py`: metadata-preserving word chunker.
- `src/embeddings.py`: sentence-transformer implementation plus deterministic test embedding model.
- `src/vector_store.py`: Chroma vector store plus in-memory store for tests.
- `src/llm_client.py`: Ollama, OpenAI-compatible, and mock LLM clients behind one interface.
- `src/prompts.py`: prompt builders for RAG, summaries, comparisons, and insufficient evidence.
- `src/summarizer.py`: structured single-paper summary workflow.
- `src/paper_compare.py`: structured multi-paper comparison workflow.
- `src/rag_pipeline.py`: retrieval plus evidence-constrained answer generation.
- `src/evaluation.py`: lightweight retrieval and answer quality checks.
- `docs/system_design.md`: project architecture and chunking rationale.
- `docs/prompt_design.md`: prompts used by the MVP.
- `docs/model_card.md`: model/provider assumptions and limitations.
- `reports/demo_queries.md`: sample queries.
- `reports/evaluation_report.md`: lightweight evaluation write-up.
- `reports/system_limitations.md`: risks, limitations, and future work.
- `data/README.md`: local data policy.
- `data/papers/.gitkeep`: keep paper directory.
- `vectorstore/.gitkeep`: keep vector store directory.
- `tests/`: focused unit tests for core behavior.

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `app/.gitkeep`
- Create: `data/README.md`
- Create: `data/papers/.gitkeep`
- Create: `vectorstore/.gitkeep`
- Create: `reports/.gitkeep`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add package metadata and dependencies**

Write `pyproject.toml` with:

```toml
[project]
name = "ai-research-assistant"
version = "0.1.0"
description = "Local-first AI research assistant for literature search, PDF ingestion, and evidence-based RAG Q&A."
requires-python = ">=3.10"
dependencies = [
  "arxiv>=2.1.3",
  "chromadb>=0.5.0",
  "pandas>=2.2.0",
  "PyMuPDF>=1.24.0",
  "requests>=2.32.0",
  "sentence-transformers>=3.0.0",
  "streamlit>=1.36.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2.0",
  "ruff>=0.5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py310"
```

- [ ] **Step 2: Add environment example**

Write `.env.example` with:

```text
AI_RA_LLM_PROVIDER=mock
AI_RA_OLLAMA_BASE_URL=http://localhost:11434
AI_RA_OLLAMA_MODEL=llama3.1
AI_RA_OPENAI_BASE_URL=
AI_RA_OPENAI_API_KEY=
AI_RA_OPENAI_MODEL=
AI_RA_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AI_RA_CHROMA_PATH=vectorstore
```

- [ ] **Step 3: Create kept directories**

Create the listed package, data, vector store, report, and test marker files.

- [ ] **Step 4: Verify scaffold**

Run: `python -m pytest --collect-only`

Expected: pytest starts successfully and reports no tests or only collected tests from files that already exist.

---

### Task 2: Settings And Shared Models

**Files:**
- Create: `src/config.py`
- Create: `src/models.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write config test**

```python
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
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `src.config` does not exist.

- [ ] **Step 3: Implement settings and dataclasses**

`src/config.py` defines:

```python
from dataclasses import dataclass
from pathlib import Path
import os


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
```

`src/models.py` defines immutable or simple dataclasses for `PaperMetadata`, `PageText`, `TextChunk`, `RetrievedChunk`, and `RagAnswer`.

- [ ] **Step 4: Run passing test**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

---

### Task 3: Text Cleaning And Chunking

**Files:**
- Create: `src/text_cleaning.py`
- Create: `src/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write chunking tests**

```python
from src.chunking import chunk_pages
from src.models import PageText


def test_chunk_pages_preserves_metadata_and_overlap():
    words = " ".join(f"word{i}" for i in range(30))
    pages = [PageText(page_number=1, text=words)]

    chunks = chunk_pages(
        pages,
        paper_title="Test Paper",
        paper_id="paper-1",
        source_url="file://test.pdf",
        chunk_size_words=10,
        chunk_overlap_words=3,
    )

    assert len(chunks) == 4
    assert chunks[0].metadata["paper_title"] == "Test Paper"
    assert chunks[0].metadata["paper_id"] == "paper-1"
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].text.split()[:3] == chunks[0].text.split()[-3:]


def test_chunk_pages_rejects_invalid_overlap():
    pages = [PageText(page_number=1, text="one two three")]

    try:
        chunk_pages(pages, "T", "P", "S", chunk_size_words=5, chunk_overlap_words=5)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_chunking.py -v`

Expected: FAIL because chunking is not implemented.

- [ ] **Step 3: Implement cleaning and chunking**

`src/text_cleaning.py` exports `clean_text(text: str) -> str` that normalizes repeated whitespace, removes hyphenated line breaks, and strips control characters.

`src/chunking.py` exports `chunk_pages(...) -> list[TextChunk]` using a sliding word window. Each chunk metadata dict includes `paper_title`, `paper_id`, `page_number`, `chunk_id`, and `source_url`.

- [ ] **Step 4: Run passing tests**

Run: `python -m pytest tests/test_chunking.py -v`

Expected: PASS.

---

### Task 4: PDF Loading

**Files:**
- Create: `src/pdf_loader.py`
- Test: `tests/test_pdf_loader.py`

- [ ] **Step 1: Write PDF loader tests**

```python
import fitz

from src.pdf_loader import PdfLoadError, load_pdf_pages


def test_load_pdf_pages_extracts_page_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Retrieval augmented generation test paper")
    doc.save(pdf_path)
    doc.close()

    pages = load_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Retrieval augmented generation" in pages[0].text


def test_load_pdf_pages_returns_clear_error_for_missing_file(tmp_path):
    try:
        load_pdf_pages(tmp_path / "missing.pdf")
    except PdfLoadError as exc:
        assert "PDF file does not exist" in str(exc)
    else:
        raise AssertionError("Expected PdfLoadError")
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_pdf_loader.py -v`

Expected: FAIL because `src.pdf_loader` does not exist.

- [ ] **Step 3: Implement PDF loading**

Create `PdfLoadError(RuntimeError)` and `load_pdf_pages(path: str | Path) -> list[PageText]`. Use `fitz.open`, extract each page with `page.get_text("text")`, clean with `clean_text`, keep non-empty pages, and wrap exceptions with a clear message.

- [ ] **Step 4: Run passing tests**

Run: `python -m pytest tests/test_pdf_loader.py -v`

Expected: PASS.

---

### Task 5: arXiv Search

**Files:**
- Create: `src/arxiv_search.py`
- Test: `tests/test_arxiv_search.py`

- [ ] **Step 1: Write arXiv conversion tests**

```python
from datetime import datetime, timezone

from src.arxiv_search import result_to_metadata


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeResult:
    entry_id = "http://arxiv.org/abs/2401.00001v1"
    title = " A Test Paper "
    authors = [FakeAuthor("Ada Lovelace"), FakeAuthor("Alan Turing")]
    published = datetime(2024, 1, 1, tzinfo=timezone.utc)
    summary = "This paper tests metadata conversion."
    pdf_url = "http://arxiv.org/pdf/2401.00001v1"


def test_result_to_metadata_normalizes_result():
    metadata = result_to_metadata(FakeResult())

    assert metadata.paper_id == "2401.00001v1"
    assert metadata.title == "A Test Paper"
    assert metadata.authors == ["Ada Lovelace", "Alan Turing"]
    assert metadata.source_url.endswith("/2401.00001v1")
    assert metadata.pdf_url.endswith("/2401.00001v1")
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_arxiv_search.py -v`

Expected: FAIL because arXiv conversion is not implemented.

- [ ] **Step 3: Implement arXiv search**

Create `result_to_metadata`, `search_arxiv`, `papers_to_dataframe`, and `save_metadata_csv`. Use `arxiv.Search` and `arxiv.Client` for live search, but keep conversion pure so tests do not need network access.

- [ ] **Step 4: Run passing tests**

Run: `python -m pytest tests/test_arxiv_search.py -v`

Expected: PASS.

---

### Task 6: Embeddings And Retrieval Stores

**Files:**
- Create: `src/embeddings.py`
- Create: `src/vector_store.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write retrieval tests**

```python
from src.embeddings import HashEmbeddingModel
from src.models import TextChunk
from src.vector_store import InMemoryVectorStore


def test_in_memory_retrieval_returns_ranked_chunks():
    embedder = HashEmbeddingModel(dimension=16)
    store = InMemoryVectorStore(embedder)
    chunks = [
        TextChunk(text="graph retrieval answer generation", metadata={"chunk_id": "c1", "paper_title": "GraphRAG"}),
        TextChunk(text="image classification with contrastive learning", metadata={"chunk_id": "c2", "paper_title": "Vision"}),
    ]

    store.add_chunks(chunks)
    results = store.search("retrieval generation", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.metadata["chunk_id"] == "c1"
    assert results[0].score > 0
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_retrieval.py -v`

Expected: FAIL because embedding and vector store modules do not exist.

- [ ] **Step 3: Implement embedding models**

Define `EmbeddingModel` protocol, `HashEmbeddingModel`, and `SentenceTransformerEmbeddingModel`. `HashEmbeddingModel` should be deterministic and dependency-light for tests. `SentenceTransformerEmbeddingModel` should lazy-load `SentenceTransformer`.

- [ ] **Step 4: Implement vector stores**

Define `InMemoryVectorStore` with cosine similarity and `ChromaVectorStore` with persistent Chroma collections. Both expose `add_chunks(chunks)` and `search(query, top_k)`.

- [ ] **Step 5: Run passing tests**

Run: `python -m pytest tests/test_retrieval.py -v`

Expected: PASS.

---

### Task 7: LLM Client And Prompts

**Files:**
- Create: `src/llm_client.py`
- Create: `src/prompts.py`
- Create: `docs/prompt_design.md`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write prompt and mock client tests**

```python
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
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_llm_client.py -v`

Expected: FAIL because LLM and prompt modules do not exist.

- [ ] **Step 3: Implement clients and prompts**

Create `LLMClient` protocol, `MockLLMClient`, `OllamaClient`, `OpenAICompatibleClient`, and `create_llm_client(settings)`. Prompt builders return complete instructions for RAG, summary, comparison, and insufficient evidence.

- [ ] **Step 4: Document prompts**

Write `docs/prompt_design.md` with the exact prompt goals and safety rules for evidence-bound answers.

- [ ] **Step 5: Run passing tests**

Run: `python -m pytest tests/test_llm_client.py -v`

Expected: PASS.

---

### Task 8: RAG, Summary, And Comparison Workflows

**Files:**
- Create: `src/rag_pipeline.py`
- Create: `src/summarizer.py`
- Create: `src/paper_compare.py`
- Test: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Write RAG tests**

```python
from src.embeddings import HashEmbeddingModel
from src.llm_client import MockLLMClient
from src.models import TextChunk
from src.rag_pipeline import RagPipeline
from src.vector_store import InMemoryVectorStore


def test_rag_pipeline_includes_evidence_and_confidence():
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=16))
    store.add_chunks([
        TextChunk(
            text="RAG retrieves relevant chunks before generating an answer.",
            metadata={"chunk_id": "c1", "paper_title": "RAG Paper", "page_number": 2, "source_url": "file://rag.pdf"},
        )
    ])
    pipeline = RagPipeline(vector_store=store, llm_client=MockLLMClient())

    answer = pipeline.answer("How does RAG work?", top_k=1)

    assert "Answer:" in answer.to_markdown()
    assert "Evidence:" in answer.to_markdown()
    assert "RAG Paper" in answer.to_markdown()
    assert answer.confidence in {"High", "Medium", "Low"}


def test_rag_pipeline_handles_empty_retrieval():
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=16))
    pipeline = RagPipeline(vector_store=store, llm_client=MockLLMClient())

    answer = pipeline.answer("What does the paper conclude?", top_k=3)

    assert answer.confidence == "Low"
    assert "insufficient" in answer.answer.lower()
    assert answer.evidence == []
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_rag_pipeline.py -v`

Expected: FAIL because RAG workflow is not implemented.

- [ ] **Step 3: Implement workflows**

`RagPipeline.answer` retrieves chunks, builds numbered context lines, calls the LLM, and returns a `RagAnswer`. Empty retrieval returns a low-confidence insufficient-evidence answer without calling the LLM. Summary and comparison modules call prompt builders and return structured markdown.

- [ ] **Step 4: Run passing tests**

Run: `python -m pytest tests/test_rag_pipeline.py -v`

Expected: PASS.

---

### Task 9: Evaluation

**Files:**
- Create: `src/evaluation.py`
- Create: `reports/demo_queries.md`
- Create: `reports/evaluation_report.md`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write evaluation tests**

```python
from src.evaluation import answer_has_citations, compute_recall_at_k


def test_compute_recall_at_k():
    retrieved = ["paper-a", "paper-b", "paper-c"]
    expected = {"paper-b", "paper-x"}

    assert compute_recall_at_k(retrieved, expected, k=3) == 0.5


def test_answer_has_citations():
    assert answer_has_citations("Evidence:\n[1] Paper, chunk c1")
    assert not answer_has_citations("Answer without sources")
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: FAIL because evaluation helpers do not exist.

- [ ] **Step 3: Implement evaluation helpers and reports**

Implement recall and citation checks. Write ten demo questions in `reports/demo_queries.md` and a concise evaluation method in `reports/evaluation_report.md`.

- [ ] **Step 4: Run passing tests**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: PASS.

---

### Task 10: Streamlit App And Documentation

**Files:**
- Create: `app/streamlit_app.py`
- Create: `docs/system_design.md`
- Create: `docs/model_card.md`
- Modify: `README.md`
- Modify: `reports/system_limitations.md`

- [ ] **Step 1: Build Streamlit app**

Create tabs for Search Papers, Upload PDFs, Ask Questions, Compare Papers, and Evaluation. Use `st.session_state` for indexed chunks, paper metadata, and vector store instances. Use mock LLM by default when no provider is configured.

- [ ] **Step 2: Write docs**

Write system design, model card, limitations, and README sections covering overview, architecture, setup, usage, evaluation, limitations, and future work.

- [ ] **Step 3: Smoke test app import**

Run: `python -m py_compile app/streamlit_app.py`

Expected: no syntax errors.

---

### Task 11: Full Verification

**Files:**
- Modify files only if verification exposes a concrete defect.

- [ ] **Step 1: Run all tests**

Run: `python -m pytest -v`

Expected: all tests pass.

- [ ] **Step 2: Run syntax check**

Run: `python -m compileall src app`

Expected: all files compile.

- [ ] **Step 3: Run Streamlit help check**

Run: `python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501`

Expected: app starts. Stop the process after confirming startup.

- [ ] **Step 4: Commit MVP**

Run:

```bash
git add .
git commit -m "Build AI research assistant MVP"
```

Expected: commit succeeds with scaffold, source, tests, app, docs, and reports.

---

## Self-Review

- Spec coverage: the tasks cover local-first setup, arXiv search, PDF loading, cleaning, chunking, embeddings, vector retrieval, RAG answers with evidence, summary, comparison, evaluation, Streamlit UI, tests, and GitHub-facing docs.
- Scope: Semantic Scholar, Zotero, citation graphs, Docker, and embedding benchmarks remain outside the MVP and are documented as future work.
- Type consistency: shared dataclasses are introduced before modules that consume them; vector stores share `add_chunks` and `search`; LLM clients share `generate`.
- Execution choice: inline execution is selected because the project owner asked the agent to decide and avoid repeated clarification.
