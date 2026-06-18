import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RetrievalEvalResult:
    question: str
    expected_items: set[str]
    retrieved_items: list[str]
    recall_at_k: float


def compute_recall_at_k(retrieved_items: list[str], expected_items: set[str], k: int) -> float:
    if not expected_items:
        return 0.0
    retrieved_at_k = set(retrieved_items[:k])
    return len(retrieved_at_k & expected_items) / len(expected_items)


def answer_has_citations(answer_text: str) -> bool:
    evidence_section = re.search(r"Evidence:\s*(.+)", answer_text, flags=re.IGNORECASE | re.DOTALL)
    if not evidence_section:
        return False
    return bool(re.search(r"\[\d+\]", evidence_section.group(1)))


def answer_claims_insufficient_evidence(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return "insufficient" in lowered or "cannot confirm" in lowered or "not enough" in lowered


def evaluate_retrieval(
    question: str,
    retrieved_items: Iterable[str],
    expected_items: Iterable[str],
    k: int = 5,
) -> RetrievalEvalResult:
    retrieved_list = list(retrieved_items)
    expected_set = set(expected_items)
    return RetrievalEvalResult(
        question=question,
        expected_items=expected_set,
        retrieved_items=retrieved_list,
        recall_at_k=compute_recall_at_k(retrieved_list, expected_set, k),
    )
