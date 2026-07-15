from pathlib import Path

import pdfplumber


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extracts text from every page in reading order.
    No cleaning or parser-specific processing.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """
    Returns one string per page.
    Useful when parsers need page-aware processing
    (e.g. removing TOC or headers).
    """
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def remove_repeated_headers(pages: list[str], threshold: float = 0.5) -> list[str]:
    """
    Removes lines repeated on many pages (running headers/footers).
    Parser decides whether to use this.
    """
    if len(pages) < 3:
        return pages

    counts = {}

    for page in pages:
        for line in page.splitlines():
            line = line.strip()
            if line:
                counts[line] = counts.get(line, 0) + 1

    repeated = {
        line
        for line, count in counts.items()
        if count >= len(pages) * threshold
    }

    cleaned = []

    for page in pages:
        kept = [
            line
            for line in page.splitlines()
            if line.strip() not in repeated
        ]
        cleaned.append("\n".join(kept))

    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Collapses excessive whitespace while preserving paragraph breaks.
    """
    import re

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()