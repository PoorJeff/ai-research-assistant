from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str]
    published: datetime | None
    summary: str
    source_url: str
    pdf_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": ", ".join(self.authors),
            "published": self.published.isoformat() if self.published else "",
            "summary": self.summary,
            "source_url": self.source_url,
            "pdf_url": self.pdf_url,
        }


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    evidence: list[str]
    confidence: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)

    def to_markdown(self) -> str:
        evidence_lines = "\n".join(self.evidence) if self.evidence else "No supporting evidence found."
        return f"Answer:\n{self.answer}\n\nEvidence:\n{evidence_lines}\n\nConfidence:\n{self.confidence}"
