from src.llm_client import LLMClient
from src.prompts import build_comparison_prompt


COMPARISON_COLUMNS = [
    "paper title",
    "research problem",
    "method",
    "dataset",
    "strength",
    "limitation",
    "possible extension",
]


def compare_papers(paper_summaries: list[str], llm_client: LLMClient) -> str:
    non_empty = [summary.strip() for summary in paper_summaries if summary.strip()]
    if not non_empty:
        return _empty_comparison_table()
    return llm_client.generate(build_comparison_prompt(non_empty)).strip()


def _empty_comparison_table() -> str:
    header = "| " + " | ".join(COMPARISON_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in COMPARISON_COLUMNS) + " |"
    return "\n".join([header, separator])
