"""
decides per page whether to use native extraction or surya OCR.
also returns the native spans directly so pipeline.py doesn't
have to re-open the pdf and re-extract what we already have.
"""

import pdfplumber
from itertools import groupby

from ocr.tokens import LineSpan
from ocr.native_extractor import NativeExtractor


def route(pdf_path: str, min_chars_per_page: int = 20) -> tuple[list[LineSpan], list[int]]:
    """
    returns (native_spans, scanned_pages)
    native_spans  -> already extracted LineSpans from pages with a text layer
    scanned_pages -> page numbers that need surya OCR
    """
    native = NativeExtractor()

    # extract everything native can get in one pass
    all_spans = native.extract(pdf_path)

    # group spans by page to check which pages have enough text
    spans_by_page: dict[int, list[LineSpan]] = {}
    for span in all_spans:
        spans_by_page.setdefault(span.page_no, []).append(span)

    # figure out total page count from the pdf directly
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    native_spans = []
    scanned_pages = []

    for page_no in range(total_pages):
        page_spans = spans_by_page.get(page_no, [])
        char_count = sum(len(s.text) for s in page_spans)

        if char_count >= min_chars_per_page:
            native_spans.extend(page_spans)
        else:
            scanned_pages.append(page_no)

    return native_spans, scanned_pages