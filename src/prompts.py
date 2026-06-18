INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I cannot confirm the answer from the current paper library because the retrieved "
    "context is insufficient."
)


def build_rag_prompt(question: str, context_lines: list[str]) -> str:
    context = "\n".join(context_lines) if context_lines else "No context was retrieved."
    return f"""You are an AI research assistant answering questions about academic papers.

Only answer based on the provided context.
Always cite evidence using the provided source numbers.
If the context is insufficient, say so clearly.
Do not invent sources, citations, paper claims, datasets, or results.

Question:
{question}

Context:
{context}

Return exactly this structure:
Answer:
...

Evidence:
[1] ...

Confidence:
High / Medium / Low
"""


def build_summary_prompt(paper_title: str, context: str) -> str:
    return f"""Summarize the paper using only the provided text.

If a field is not clearly supported, write "Not clearly stated in the provided text".

Paper title:
{paper_title}

Text:
{context}

Return these fields:
- Research Problem
- Method
- Dataset / Experiment
- Key Findings
- Limitations
- Potential Applications
"""


def build_comparison_prompt(paper_contexts: list[str]) -> str:
    joined = "\n\n".join(paper_contexts)
    return f"""Compare the papers using only the provided paper summaries or excerpts.

Do not invent methods, datasets, strengths, or limitations.
If a field is not supported, write "Not clearly stated in the provided text".

Papers:
{joined}

Return a Markdown table with columns:
paper title | research problem | method | dataset | strength | limitation | possible extension
"""


def build_insufficient_evidence_response() -> str:
    return f"""Answer:
{INSUFFICIENT_EVIDENCE_MESSAGE}

Evidence:
No supporting evidence found.

Confidence:
Low
"""
