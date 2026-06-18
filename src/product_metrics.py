from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.real_paper_evaluation import PaperRunRecord, QueryEvaluationResult, RealPaperEvaluationResult


@dataclass(frozen=True)
class EvaluationSummary:
    paper_count: int
    page_count: int
    chunk_count: int
    question_count: int
    average_recall_at_3: float
    average_recall_at_5: float
    citation_coverage_count: int


def load_real_paper_evaluation(
    report_path: Path = Path("reports/real_paper_evaluation.json"),
) -> RealPaperEvaluationResult | None:
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return parse_real_paper_evaluation(data)


def parse_real_paper_evaluation(data: dict[str, Any]) -> RealPaperEvaluationResult:
    return RealPaperEvaluationResult(
        generated_at=str(data["generated_at"]),
        embedding_backend=str(data["embedding_backend"]),
        llm_provider=str(data["llm_provider"]),
        papers=[PaperRunRecord(**paper) for paper in data.get("papers", [])],
        query_results=[QueryEvaluationResult(**query) for query in data.get("query_results", [])],
    )


def summarize_real_paper_evaluation(
    result: RealPaperEvaluationResult | dict[str, Any],
) -> EvaluationSummary:
    if isinstance(result, dict):
        result = parse_real_paper_evaluation(result)

    return EvaluationSummary(
        paper_count=len(result.papers),
        page_count=sum(paper.pages for paper in result.papers),
        chunk_count=sum(paper.chunks for paper in result.papers),
        question_count=len(result.query_results),
        average_recall_at_3=result.average_recall_at_3,
        average_recall_at_5=result.average_recall_at_5,
        citation_coverage_count=result.citation_coverage_count,
    )
