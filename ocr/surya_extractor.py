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

    def _render_pages(self, pdf_path: str, page_numbers: list[int], scale: float = 2.0) -> list[Image.Image]:
        """
        opens the pdf once, renders all requested pages, closes it.
        one parse of the pdf header instead of one per page.
        scale=2.0 ~= 144 DPI, recommended floor for decent OCR accuracy.
        """
        doc = pdfium.PdfDocument(pdf_path)
        images = []
        for page_no in page_numbers:
            bitmap = doc[page_no].render(scale=scale) #type:ignore
            images.append(bitmap.to_pil())
        doc.close()
        return images

    def extract(self, pdf_path: str, page_numbers: list[int], chunk_size: int = 4) -> list[LineSpan]:
        """
        runs surya only on the given page numbers.
        renders all pages with one pdf open, then feeds surya in chunks
        to avoid OOM on large scanned documents - batching all 30 pages
        at once on a 6gb card will OOM since detection holds activation
        memory for every image simultaneously.
        chunk_size=4 is conservative for 6gb vram, tune up if you have headroom.
        """
        images = self._render_pages(pdf_path, page_numbers)

        all_spans = []

        for i in range(0, len(images), chunk_size):
            chunk_images = images[i:i + chunk_size]
            chunk_page_nos = page_numbers[i:i + chunk_size]

            results = self.recognition_predictor(
                chunk_images,
                [None] * len(chunk_images),
                self.detection_predictor,
            )

            for page_no, page_result in zip(chunk_page_nos, results):
                for line in page_result.text_lines:
                    text = line.text.strip()
                    if not text:
                        continue

                    x0, y0, x1, y1 = line.bbox #type:ignore
                    span = LineSpan(
                        text=text,
                        page_no=page_no,
                        source="surya",
                        x0=x0, y0=y0, x1=x1, y1=y1,
                    )
                    if span.is_valid():
                        all_spans.append(span)

        return all_spans