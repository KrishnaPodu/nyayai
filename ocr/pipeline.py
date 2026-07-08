"""
single entrypoint for the entire OCR pipeline.
calls router which extracts native spans + identifies scanned pages
in one pass, then only calls surya on pages that actually need it.
"""

from ocr.tokens import LineSpan
from ocr.router import route


def extract(pdf_path: str, min_chars_per_page: int = 2000) -> list[LineSpan]:
    native_spans, scanned_pages = route(pdf_path, min_chars_per_page)

    spans = list(native_spans)

    if scanned_pages:
        # lazy import so surya models only load when actually needed
        from ocr.surya_extractor import SuryaExtractor
        surya = SuryaExtractor()
        spans.extend(surya.extract(pdf_path, scanned_pages))

    # sort by page then top-to-bottom reading order
    spans.sort(key=lambda s: (s.page_no, s.y0))

    return spans