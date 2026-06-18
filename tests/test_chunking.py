from src.chunking import chunk_pages
from src.models import PageText


def test_chunk_pages_preserves_metadata_and_overlap():
    words = " ".join(f"word{i}" for i in range(30))
    pages = [PageText(page_number=1, text=words)]

    chunks = chunk_pages(
        pages,
        paper_title="Test Paper",
        paper_id="paper-1",
        source_url="file://test.pdf",
        chunk_size_words=10,
        chunk_overlap_words=3,
    )

    assert len(chunks) == 4
    assert chunks[0].metadata["paper_title"] == "Test Paper"
    assert chunks[0].metadata["paper_id"] == "paper-1"
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].text.split()[:3] == chunks[0].text.split()[-3:]


def test_chunk_pages_rejects_invalid_overlap():
    pages = [PageText(page_number=1, text="one two three")]

    try:
        chunk_pages(pages, "T", "P", "S", chunk_size_words=5, chunk_overlap_words=5)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
