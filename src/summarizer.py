from src.llm_client import LLMClient
from src.models import TextChunk
from src.prompts import build_summary_prompt


SUMMARY_FIELDS = [
    "Research Problem",
    "Method",
    "Dataset / Experiment",
    "Key Findings",
    "Limitations",
    "Potential Applications",
]


def summarize_paper(
    paper_title: str,
    chunks: list[TextChunk],
    llm_client: LLMClient,
    max_chars: int = 12000,
) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)[:max_chars]
    if not context.strip():
        return _empty_summary()
    return llm_client.generate(build_summary_prompt(paper_title, context)).strip()


def _empty_summary() -> str:
    lines = ["| Field | Summary |", "| --- | --- |"]
    for field in SUMMARY_FIELDS:
        lines.append(f"| {field} | Not clearly stated in the provided text |")
    return "\n".join(lines)
