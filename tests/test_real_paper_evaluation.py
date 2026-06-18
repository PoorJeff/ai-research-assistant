from src.real_paper_evaluation import (
    PaperRunRecord,
    QueryEvaluationResult,
    RealPaperEvaluationResult,
    default_demo_questions,
    format_markdown_report,
)


def test_default_demo_questions_have_expected_papers():
    questions = default_demo_questions()

    assert len(questions) >= 5
    assert all(question.expected_paper_ids for question in questions)
    assert {"2005.11401", "2404.16130", "2309.15217"} <= {
        paper_id for question in questions for paper_id in question.expected_paper_ids
    }


def test_format_markdown_report_includes_metrics_and_papers():
    result = RealPaperEvaluationResult(
        generated_at="2026-06-18T16:30:00+08:00",
        embedding_backend="HashEmbeddingModel",
        llm_provider="mock",
        papers=[
            PaperRunRecord(
                paper_id="2005.11401",
                title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                pdf_url="https://arxiv.org/pdf/2005.11401",
                pages=10,
                chunks=20,
            )
        ],
        query_results=[
            QueryEvaluationResult(
                question="What is retrieval augmented generation?",
                expected_paper_ids=["2005.11401"],
                retrieved_paper_ids_at_3=["2005.11401"],
                retrieved_paper_ids_at_5=["2005.11401"],
                recall_at_3=1.0,
                recall_at_5=1.0,
                citation_present=True,
                confidence="Medium",
                evidence_count=1,
            )
        ],
    )

    markdown = format_markdown_report(result)

    assert "Average Recall@3: 1.00" in markdown
    assert "Citation coverage: 1/1" in markdown
    assert "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" in markdown
