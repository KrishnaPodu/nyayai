"""
decides per page whether to use native extraction or surya OCR.
pipeline.py calls this to split pages into two lists before doing
any actual extraction.
"""

from ocr.native_extractor import NativeExtractor


def route(pdf_path: str, min_chars_per_page: int = 20) -> tuple[list[int], list[int]]:
    """
    returns (native_pages, scanned_pages) - two lists of page numbers.
    native_pages  -> have a usable text layer, use NativeExtractor
    scanned_pages -> no text layer, need surya OCR
    """
    native = NativeExtractor()
    text_layer = native.has_text_layer(pdf_path, min_chars_per_page)

    native_pages = [p for p, has_text in text_layer.items() if has_text]
    scanned_pages = [p for p, has_text in text_layer.items() if not has_text]

    return native_pages, scanned_pages