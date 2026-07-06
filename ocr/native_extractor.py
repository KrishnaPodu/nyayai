"""
extracts text from pdfs that already have a text layer.

pdfplumber gives word-level boxes, so we group words that share the
same vertical position (top) into lines. the line's bbox spans from
the leftmost word's x0 to the rightmost word's x1.
"""

import pdfplumber
from itertools import groupby

from ocr.tokens import LineSpan


class NativeExtractor:
    def extract(self, pdf_path: str) -> list[LineSpan]:
        spans = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                words = page.extract_words()
                spans.extend(self._words_to_linespans(words, page_no))

        return spans

    def _words_to_linespans(self, words: list[dict], page_no: int) -> list[LineSpan]:
        if not words:
            return []

        # group words by their top value - words on the same line share
        # the same top coordinate in pdfplumber
        # round to 1 decimal to absorb tiny floating point jitter between
        # words on the same visual line
        words_sorted = sorted(words, key=lambda w: round(w["top"], 1))

        spans = []
        for top_val, group in groupby(words_sorted, key=lambda w: round(w["top"], 1)):
            line_words = list(group)

            text = " ".join(w["text"] for w in line_words)
            x0 = min(w["x0"] for w in line_words)
            x1 = max(w["x1"] for w in line_words)
            y0 = min(w["top"] for w in line_words)
            y1 = max(w["bottom"] for w in line_words)

            span = LineSpan(
                text=text,
                page_no=page_no,
                source="native",
                x0=x0, y0=y0, x1=x1, y1=y1,
            )
            if span.is_valid():
                spans.append(span)

        return spans

    def has_text_layer(self, pdf_path: str, min_chars_per_page: int = 20) -> dict[int, bool]:
        """
        per page check - does this page have enough native text to skip OCR?
        router.py uses this to decide which pages to send to surya.
        """
        result = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                result[page_no] = len(text.strip()) >= min_chars_per_page

        return result