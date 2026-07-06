"""
single entrypoint for the entire OCR pipeline.
calls router to split pages, then runs the right extractor on each.
surya is only loaded if at least one page needs it.
"""

from ocr.tokens import LineSpan
from ocr.native_extractor import NativeExtractor
from ocr.router import route


def extract(pdf_path: str, min_chars_per_page: int = 20) -> list[LineSpan]:
    native_pages, scanned_pages = route(pdf_path, min_chars_per_page)

    spans = []

    if native_pages:
        native = NativeExtractor()
        all_spans = native.extract(pdf_path)
        spans.extend(s for s in all_spans if s.page_no in set(native_pages))

    if scanned_pages:
        # lazy import so surya models only load when actually needed
        from ocr.surya_extractor import SuryaExtractor
        surya = SuryaExtractor()
        spans.extend(surya.extract(pdf_path, scanned_pages))

    # sort by page then top-to-bottom reading order
    spans.sort(key=lambda s: (s.page_no, s.y0))

    return spans