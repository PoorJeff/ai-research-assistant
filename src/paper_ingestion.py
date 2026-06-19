from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import re

import requests

from src.chunking import chunk_pages
from src.models import PageText, PaperMetadata, TextChunk
from src.pdf_loader import load_pdf_pages
from src.real_paper_evaluation import normalize_arxiv_id


class PaperIngestionError(RuntimeError):
    """Raised when a paper PDF cannot be downloaded or prepared."""


@dataclass(frozen=True)
class PaperIngestionRecord:
    paper_id: str
    title: str
    pdf_path: str
    pages: int
    chunks: int
    source_url: str
    pdf_url: str


Downloader = Callable[[PaperMetadata, Path], Path]
PageLoader = Callable[[Path], list[PageText]]


def paper_pdf_filename(paper: PaperMetadata) -> str:
    paper_id = normalize_arxiv_id(paper.paper_id)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id).strip("-")
    if safe_id:
        return f"{safe_id}.pdf"
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "-", paper.title.lower()).strip("-")
    return f"{safe_title or 'paper'}.pdf"


def download_paper_pdf(paper: PaperMetadata, output_dir: Path, timeout: int = 90) -> Path:
    if not paper.pdf_url:
        raise PaperIngestionError(f"No PDF URL available for paper: {paper.title}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / paper_pdf_filename(paper)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    try:
        response = requests.get(paper.pdf_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PaperIngestionError(f"Failed to download PDF for {paper.title}: {exc}") from exc

    if not response.content:
        raise PaperIngestionError(f"Downloaded PDF is empty for paper: {paper.title}")

    output_path.write_bytes(response.content)
    return output_path


def ingest_papers_from_search(
    papers: Iterable[PaperMetadata],
    output_dir: Path = Path("data/papers/imported"),
    chunk_size_words: int = 700,
    chunk_overlap_words: int = 120,
    downloader: Downloader = download_paper_pdf,
    page_loader: PageLoader = load_pdf_pages,
) -> tuple[list[TextChunk], list[PaperIngestionRecord]]:
    all_chunks: list[TextChunk] = []
    records: list[PaperIngestionRecord] = []

    for paper in papers:
        paper_id = normalize_arxiv_id(paper.paper_id)
        pdf_path = downloader(paper, output_dir)
        pages = page_loader(pdf_path)
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
            PaperIngestionRecord(
                paper_id=paper_id,
                title=paper.title,
                pdf_path=str(pdf_path),
                pages=len(pages),
                chunks=len(chunks),
                source_url=paper.source_url,
                pdf_url=paper.pdf_url,
            )
        )

    return all_chunks, records
