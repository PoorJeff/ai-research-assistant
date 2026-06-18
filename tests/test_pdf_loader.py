import fitz

from src.pdf_loader import PdfLoadError, load_pdf_pages


def test_load_pdf_pages_extracts_page_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Retrieval augmented generation test paper")
    doc.save(pdf_path)
    doc.close()

    pages = load_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Retrieval augmented generation" in pages[0].text


def test_load_pdf_pages_returns_clear_error_for_missing_file(tmp_path):
    try:
        load_pdf_pages(tmp_path / "missing.pdf")
    except PdfLoadError as exc:
        assert "PDF file does not exist" in str(exc)
    else:
        raise AssertionError("Expected PdfLoadError")
