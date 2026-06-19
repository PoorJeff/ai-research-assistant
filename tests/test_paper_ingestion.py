from datetime import datetime
from pathlib import Path

import pytest

from src.models import PageText, PaperMetadata
from src.paper_ingestion import (
    PaperIngestionError,
    download_paper_pdf,
    ingest_papers_from_search,
    paper_pdf_filename,
)


def make_paper(paper_id: str = "2309.15217v2", pdf_url: str = "https://example.com/paper.pdf") -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title="Ragas: Automated Evaluation of Retrieval Augmented Generation",
        authors=["Researcher A"],
        published=datetime(2023, 9, 1),
        summary="Evaluation paper.",
        source_url="https://arxiv.org/abs/2309.15217",
        pdf_url=pdf_url,
    )


def test_paper_pdf_filename_normalizes_versioned_arxiv_ids():
    assert paper_pdf_filename(make_paper()) == "2309.15217.pdf"


def test_download_paper_pdf_reuses_existing_non_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paper = make_paper()
    existing_pdf = tmp_path / "2309.15217.pdf"
    existing_pdf.write_bytes(b"%PDF cached")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("download should not run when cached PDF exists")

    monkeypatch.setattr("src.paper_ingestion.requests.get", fail_if_called)

    assert download_paper_pdf(paper, tmp_path) == existing_pdf


def test_download_paper_pdf_requires_pdf_url(tmp_path: Path):
    with pytest.raises(PaperIngestionError, match="No PDF URL"):
        download_paper_pdf(make_paper(pdf_url=""), tmp_path)


def test_ingest_papers_from_search_downloads_loads_and_chunks(tmp_path: Path):
    paper = make_paper()
    fake_pdf = tmp_path / "2309.15217.pdf"
    fake_pdf.write_bytes(b"%PDF")

    def downloader(_paper: PaperMetadata, output_dir: Path) -> Path:
        assert output_dir == tmp_path
        return fake_pdf

    def page_loader(path: Path) -> list[PageText]:
        assert path == fake_pdf
        return [
            PageText(
                page_number=1,
                text="RAGAS evaluates retrieval augmented generation with faithfulness and context metrics.",
            )
        ]

    chunks, records = ingest_papers_from_search(
        [paper],
        output_dir=tmp_path,
        chunk_size_words=80,
        chunk_overlap_words=0,
        downloader=downloader,
        page_loader=page_loader,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["paper_id"] == "2309.15217"
    assert chunks[0].metadata["paper_title"] == paper.title
    assert chunks[0].metadata["source_url"] == paper.source_url
    assert records[0].paper_id == "2309.15217"
    assert records[0].pages == 1
    assert records[0].chunks == 1
