from pathlib import Path

import fitz

from src.models import PageText
from src.text_cleaning import clean_text


class PdfLoadError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


def load_pdf_pages(path: str | Path) -> list[PageText]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise PdfLoadError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise PdfLoadError(f"PDF path is not a file: {pdf_path}")

    pages: list[PageText] = []
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count == 0:
                raise PdfLoadError(f"PDF has no pages: {pdf_path}")
            for index, page in enumerate(document, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append(PageText(page_number=index, text=text))
    except PdfLoadError:
        raise
    except Exception as exc:
        raise PdfLoadError(f"Failed to parse PDF {pdf_path}: {exc}") from exc

    if not pages:
        raise PdfLoadError(f"No extractable text found in PDF: {pdf_path}")
    return pages
