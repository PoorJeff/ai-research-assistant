import re
import unicodedata


_CONTROL_CHARS = dict.fromkeys(
    char for char in range(32) if chr(char) not in {"\n", "\t", "\r"}
)


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.translate(_CONTROL_CHARS)
    normalized = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s*\n\s*", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]*([.,;:!?])", r"\1", normalized)
    return normalized.strip()


def clean_for_chunking(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()
