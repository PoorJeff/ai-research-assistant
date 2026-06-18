from datetime import datetime, timezone

from src.arxiv_search import result_to_metadata


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeResult:
    entry_id = "http://arxiv.org/abs/2401.00001v1"
    title = " A Test Paper "
    authors = [FakeAuthor("Ada Lovelace"), FakeAuthor("Alan Turing")]
    published = datetime(2024, 1, 1, tzinfo=timezone.utc)
    summary = "This paper tests metadata conversion."
    pdf_url = "http://arxiv.org/pdf/2401.00001v1"


def test_result_to_metadata_normalizes_result():
    metadata = result_to_metadata(FakeResult())

    assert metadata.paper_id == "2401.00001v1"
    assert metadata.title == "A Test Paper"
    assert metadata.authors == ["Ada Lovelace", "Alan Turing"]
    assert metadata.source_url.endswith("/2401.00001v1")
    assert metadata.pdf_url.endswith("/2401.00001v1")
