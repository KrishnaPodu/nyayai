"""
extracts text from scanned/image-only pdfs using surya OCR.

surya natively gives line-level text + line-level bboxes which maps
directly to LineSpan - no approximation needed anymore. this is the
whole reason we switched from WordToken to LineSpan.
"""

import pypdfium2 as pdfium
from PIL import Image

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from ocr.tokens import LineSpan


class SuryaExtractor:
    def __init__(self):
        # load once, reuse across all pages - don't reinstantiate per page
        self.detection_predictor = DetectionPredictor()
        self.recognition_predictor = RecognitionPredictor()

    def _render_page(self, pdf_path: str, page_no: int, scale: float = 2.0) -> Image.Image:
        """
        rasterizes one pdf page to a PIL image for surya.
        scale=2.0 ~= 144 DPI, recommended floor for decent OCR accuracy.
        """
        doc = pdfium.PdfDocument(pdf_path)
        page = doc[page_no]
        bitmap = page.render(scale=scale) #type: ignore
        image = bitmap.to_pil()
        doc.close()
        return image

    def extract_page(self, pdf_path: str, page_no: int) -> list[LineSpan]:
        image = self._render_page(pdf_path, page_no)

        results = self.recognition_predictor(
            [image],
            [None],   # None = auto detect language, recommended in 0.9.3
            self.detection_predictor,
        )

        spans = []
        for line in results[0].text_lines:
            text = line.text.strip()
            if not text:
                continue

            x0, y0, x1, y1 = line.bbox #type: ignore
            span = LineSpan(
                text=text,
                page_no=page_no,
                source="surya",
                x0=x0, y0=y0, x1=x1, y1=y1,
            )
            if span.is_valid():
                spans.append(span)

        return spans

    def extract(self, pdf_path: str, page_numbers: list[int]) -> list[LineSpan]:
        """
        runs surya only on the given page numbers.
        router.py passes in only the pages that need OCR.
        """
        all_spans = []
        for page_no in page_numbers:
            all_spans.extend(self.extract_page(pdf_path, page_no))
        return all_spans