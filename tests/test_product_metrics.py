from pathlib import Path

from src.product_metrics import (
    EvaluationSummary,
    load_real_paper_evaluation,
    summarize_real_paper_evaluation,
)


def test_load_real_paper_evaluation_from_json(tmp_path: Path):
    report_path = tmp_path / "real_paper_evaluation.json"
    report_path.write_text(
        """
{
  "generated_at": "2026-06-18T08:38:54.212949+00:00",
  "embedding_backend": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_provider": "mock",
  "papers": [
    {
      "paper_id": "2005.11401",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "pdf_url": "https://arxiv.org/pdf/2005.11401v4",
      "pages": 19,
      "chunks": 20
    }
  ],
  "query_results": [
    {
      "question": "Which paper introduced RAG?",
      "expected_paper_ids": ["2005.11401"],
      "retrieved_paper_ids_at_3": ["2005.11401"],
      "retrieved_paper_ids_at_5": ["2005.11401"],
      "recall_at_3": 1.0,
      "recall_at_5": 1.0,
      "citation_present": true,
      "confidence": "High",
      "evidence_count": 5
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    result = load_real_paper_evaluation(report_path)

    assert result is not None
    assert result.embedding_backend == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.papers[0].paper_id == "2005.11401"
    assert result.query_results[0].citation_present is True


def test_load_real_paper_evaluation_returns_none_for_missing_file(tmp_path: Path):
    assert load_real_paper_evaluation(tmp_path / "missing.json") is None


def test_summarize_real_paper_evaluation_counts_corpus_and_metrics():
    summary = summarize_real_paper_evaluation(
        {
            "generated_at": "2026-06-18T08:38:54.212949+00:00",
            "embedding_backend": "HashEmbeddingModel",
            "llm_provider": "mock",
            "papers": [
                {"paper_id": "a", "title": "Paper A", "pdf_url": "https://example.com/a.pdf", "pages": 10, "chunks": 12},
                {"paper_id": "b", "title": "Paper B", "pdf_url": "https://example.com/b.pdf", "pages": 8, "chunks": 9},
            ],
            "query_results": [
                {
                    "question": "Question A",
                    "expected_paper_ids": ["a"],
                    "retrieved_paper_ids_at_3": ["a"],
                    "retrieved_paper_ids_at_5": ["a"],
                    "recall_at_3": 1.0,
                    "recall_at_5": 1.0,
                    "citation_present": True,
                    "confidence": "High",
                    "evidence_count": 3,
                },
                {
                    "question": "Question B",
                    "expected_paper_ids": ["b"],
                    "retrieved_paper_ids_at_3": ["x"],
                    "retrieved_paper_ids_at_5": ["b"],
                    "recall_at_3": 0.0,
                    "recall_at_5": 1.0,
                    "citation_present": False,
                    "confidence": "Low",
                    "evidence_count": 1,
                },
            ],
        }
    )

    assert summary == EvaluationSummary(
        paper_count=2,
        page_count=18,
        chunk_count=21,
        question_count=2,
        average_recall_at_3=0.5,
        average_recall_at_5=1.0,
        citation_coverage_count=1,
    )
