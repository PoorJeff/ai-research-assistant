from src.evaluation import answer_has_citations, compute_recall_at_k


def test_compute_recall_at_k():
    retrieved = ["paper-a", "paper-b", "paper-c"]
    expected = {"paper-b", "paper-x"}

    assert compute_recall_at_k(retrieved, expected, k=3) == 0.5


def test_answer_has_citations():
    assert answer_has_citations("Evidence:\n[1] Paper, chunk c1")
    assert not answer_has_citations("Answer without sources")
