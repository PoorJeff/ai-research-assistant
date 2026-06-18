from collections.abc import Iterable

from src.models import PageText, TextChunk
from src.text_cleaning import clean_for_chunking


def chunk_pages(
    pages: Iterable[PageText],
    paper_title: str,
    paper_id: str,
    source_url: str,
    chunk_size_words: int = 700,
    chunk_overlap_words: int = 120,
) -> list[TextChunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must be non-negative")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[TextChunk] = []
    chunk_index = 0
    step = chunk_size_words - chunk_overlap_words

    for page in pages:
        words = clean_for_chunking(page.text).split()
        if not words:
            continue

        for start in range(0, len(words), step):
            window = words[start : start + chunk_size_words]
            if not window:
                continue

            chunk_id = f"{paper_id}-p{page.page_number}-c{chunk_index}"
            chunks.append(
                TextChunk(
                    text=" ".join(window),
                    metadata={
                        "paper_title": paper_title,
                        "paper_id": paper_id,
                        "page_number": page.page_number,
                        "chunk_id": chunk_id,
                        "source_url": source_url,
                    },
                )
            )
            chunk_index += 1

            if start + chunk_size_words >= len(words):
                break

    return chunks
