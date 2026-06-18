from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.models import PaperMetadata


def _paper_id_from_entry(entry_id: str) -> str:
    return entry_id.rstrip("/").split("/")[-1]


def result_to_metadata(result: Any) -> PaperMetadata:
    source_url = str(result.entry_id).strip()
    return PaperMetadata(
        paper_id=_paper_id_from_entry(source_url),
        title=" ".join(str(result.title).split()),
        authors=[author.name for author in getattr(result, "authors", [])],
        published=getattr(result, "published", None),
        summary=" ".join(str(getattr(result, "summary", "")).split()),
        source_url=source_url,
        pdf_url=str(getattr(result, "pdf_url", "")).strip(),
    )


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> list[PaperMetadata]:
    import arxiv

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
        "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
    }
    criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=criterion,
    )
    client = arxiv.Client()
    return [result_to_metadata(result) for result in client.results(search)]


def papers_to_dataframe(papers: Iterable[PaperMetadata]) -> pd.DataFrame:
    return pd.DataFrame([paper.to_dict() for paper in papers])


def save_metadata_csv(papers: Iterable[PaperMetadata], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    papers_to_dataframe(papers).to_csv(output_path, index=False)
    return output_path
