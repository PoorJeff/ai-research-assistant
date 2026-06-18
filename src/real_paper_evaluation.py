from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import requests

from src.arxiv_search import result_to_metadata
from src.chunking import chunk_pages
from src.embeddings import HashEmbeddingModel, SentenceTransformerEmbeddingModel
from src.evaluation import answer_has_citations, compute_recall_at_k
from src.llm_client import MockLLMClient
from src.models import PaperMetadata, TextChunk
from src.pdf_loader import load_pdf_pages
from src.rag_pipeline import RagPipeline
from src.vector_store import InMemoryVectorStore


DEFAULT_PAPER_IDS = ["2005.11401", "2404.16130", "2309.15217"]


@dataclass(frozen=True)
class DemoQuestion:
    question: str
    expected_paper_ids: list[str]
    rationale: str


@dataclass(frozen=True)
class PaperRunRecord:
    paper_id: str
    title: str
    pdf_url: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class QueryEvaluationResult:
    question: str
    expected_paper_ids: list[str]
    retrieved_paper_ids_at_3: list[str]
    retrieved_paper_ids_at_5: list[str]
    recall_at_3: float
    recall_at_5: float
    citation_present: bool
    confidence: str
    evidence_count: int


@dataclass(frozen=True)
class RealPaperEvaluationResult:
    generated_at: str
    embedding_backend: str
    llm_provider: str
    papers: list[PaperRunRecord]
    query_results: list[QueryEvaluationResult]

    @property
    def average_recall_at_3(self) -> float:
        return _average(result.recall_at_3 for result in self.query_results)

    @property
    def average_recall_at_5(self) -> float:
        return _average(result.recall_at_5 for result in self.query_results)

    @property
    def citation_coverage_count(self) -> int:
        return sum(1 for result in self.query_results if result.citation_present)


def normalize_arxiv_id(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id.strip())


def default_demo_questions() -> list[DemoQuestion]:
    return [
        DemoQuestion(
            question="What is the main idea behind retrieval augmented generation?",
            expected_paper_ids=["2005.11401"],
            rationale="The original RAG paper defines retrieval-augmented generation.",
        ),
        DemoQuestion(
            question="How does GraphRAG use graph structure for global sensemaking?",
            expected_paper_ids=["2404.16130"],
            rationale="The GraphRAG paper focuses on graph-based retrieval and global summaries.",
        ),
        DemoQuestion(
            question="What does RAGAS evaluate in retrieval augmented generation systems?",
            expected_paper_ids=["2309.15217"],
            rationale="The RAGAS paper focuses on evaluating RAG pipelines.",
        ),
        DemoQuestion(
            question="Which paper discusses answer faithfulness and citation-style evaluation?",
            expected_paper_ids=["2309.15217"],
            rationale="RAGAS includes answer-level faithfulness and context quality measures.",
        ),
        DemoQuestion(
            question="Which paper introduced retrieval-augmented generation for knowledge-intensive NLP?",
            expected_paper_ids=["2005.11401"],
            rationale="This query should retrieve the Lewis et al. RAG paper.",
        ),
        DemoQuestion(
            question="Which paper explains graph-based retrieval augmented generation?",
            expected_paper_ids=["2404.16130"],
            rationale="This query should retrieve the GraphRAG paper.",
        ),
    ]


def fetch_arxiv_metadata(paper_ids: list[str]) -> list[PaperMetadata]:
    import arxiv

    search = arxiv.Search(id_list=paper_ids)
    client = arxiv.Client()
    papers = [result_to_metadata(result) for result in client.results(search)]
    return sorted(papers, key=lambda paper: paper_ids.index(normalize_arxiv_id(paper.paper_id)))


def download_pdf(paper: PaperMetadata, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_id = normalize_arxiv_id(paper.paper_id)
    output_path = output_dir / f"{paper_id}.pdf"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    response = requests.get(paper.pdf_url, timeout=90)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def build_chunks_for_papers(
    papers: list[PaperMetadata],
    pdf_dir: Path,
    chunk_size_words: int = 700,
    chunk_overlap_words: int = 120,
) -> tuple[list[TextChunk], list[PaperRunRecord]]:
    all_chunks: list[TextChunk] = []
    records: list[PaperRunRecord] = []
    for paper in papers:
        pdf_path = download_pdf(paper, pdf_dir)
        pages = load_pdf_pages(pdf_path)
        paper_id = normalize_arxiv_id(paper.paper_id)
        chunks = chunk_pages(
            pages,
            paper_title=paper.title,
            paper_id=paper_id,
            source_url=paper.source_url,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
        )
        all_chunks.extend(chunks)
        records.append(
            PaperRunRecord(
                paper_id=paper_id,
                title=paper.title,
                pdf_url=paper.pdf_url,
                pages=len(pages),
                chunks=len(chunks),
            )
        )
    return all_chunks, records


def run_real_paper_evaluation(
    paper_ids: list[str] | None = None,
    pdf_dir: Path = Path("data/papers/real_demo"),
    embedding_backend: str = "sentence-transformer",
) -> RealPaperEvaluationResult:
    selected_ids = paper_ids or DEFAULT_PAPER_IDS
    papers = fetch_arxiv_metadata(selected_ids)
    chunks, paper_records = build_chunks_for_papers(papers, pdf_dir)

    if embedding_backend == "sentence-transformer":
        embedder = SentenceTransformerEmbeddingModel()
        embedding_name = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        embedder = HashEmbeddingModel(dimension=384)
        embedding_name = "HashEmbeddingModel"

    store = InMemoryVectorStore(embedder)
    store.add_chunks(chunks)
    pipeline = RagPipeline(store, MockLLMClient())

    query_results: list[QueryEvaluationResult] = []
    for demo_question in default_demo_questions():
        answer = pipeline.answer(demo_question.question, top_k=5)
        retrieved_ids = [
            normalize_arxiv_id(str(result.chunk.metadata.get("paper_id", "")))
            for result in answer.retrieved_chunks
        ]
        query_results.append(
            QueryEvaluationResult(
                question=demo_question.question,
                expected_paper_ids=demo_question.expected_paper_ids,
                retrieved_paper_ids_at_3=retrieved_ids[:3],
                retrieved_paper_ids_at_5=retrieved_ids[:5],
                recall_at_3=compute_recall_at_k(
                    retrieved_ids,
                    set(demo_question.expected_paper_ids),
                    k=3,
                ),
                recall_at_5=compute_recall_at_k(
                    retrieved_ids,
                    set(demo_question.expected_paper_ids),
                    k=5,
                ),
                citation_present=answer_has_citations(answer.to_markdown()),
                confidence=answer.confidence,
                evidence_count=len(answer.evidence),
            )
        )

    return RealPaperEvaluationResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        embedding_backend=embedding_name,
        llm_provider="mock",
        papers=paper_records,
        query_results=query_results,
    )


def format_markdown_report(result: RealPaperEvaluationResult) -> str:
    lines = [
        "# Real Paper Evaluation",
        "",
        f"Generated at: `{result.generated_at}`",
        "",
        "## Configuration",
        "",
        f"- Embedding backend: `{result.embedding_backend}`",
        f"- LLM provider: `{result.llm_provider}`",
        "- Vector store: `InMemoryVectorStore` using the same retrieval interface as the app",
        "- Corpus source: real arXiv PDFs parsed locally with PyMuPDF",
        "",
        "## Corpus",
        "",
        "| paper id | title | pages | chunks |",
        "| --- | --- | ---: | ---: |",
    ]
    for paper in result.papers:
        lines.append(f"| {paper.paper_id} | {paper.title} | {paper.pages} | {paper.chunks} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- Average Recall@3: {result.average_recall_at_3:.2f}",
            f"- Average Recall@5: {result.average_recall_at_5:.2f}",
            f"- Citation coverage: {result.citation_coverage_count}/{len(result.query_results)}",
            "",
            "## Query Results",
            "",
            "| question | expected papers | retrieved @3 | Recall@3 | Recall@5 | citations | confidence |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for query in result.query_results:
        citation_text = "yes" if query.citation_present else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    query.question,
                    ", ".join(query.expected_paper_ids),
                    ", ".join(query.retrieved_paper_ids_at_3),
                    f"{query.recall_at_3:.2f}",
                    f"{query.recall_at_5:.2f}",
                    citation_text,
                    query.confidence,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run evaluates retrieval and citation formatting on real PDF text. "
            "The generation provider is the deterministic `mock` client so the reported "
            "answer-quality metric is citation coverage rather than semantic answer quality. "
            "Use Ollama or an OpenAI-compatible provider for qualitative answer evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evaluation_artifacts(
    result: RealPaperEvaluationResult,
    markdown_path: Path = Path("reports/real_paper_evaluation.md"),
    json_path: Path = Path("reports/real_paper_evaluation.json"),
) -> None:
    markdown_path.write_text(format_markdown_report(result), encoding="utf-8")
    json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def _average(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
