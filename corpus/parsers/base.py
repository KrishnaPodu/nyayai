"""
shared parsing infrastructure.

BaseParser is the interface every act-specific parser implements: one
`parse(pdf_path) -> list[Section]` method, nothing else (no chunking, no
embedding, no uploading - those are separate stages of the pipeline).

ChapterSectionParser implements the Chapter -> Section grammar shared by
IPC/BNS/BNSS/CPC. only the Constitution (Part -> Chapter -> Article) needs
a fully separate implementation, in parsers/constitution.py.
"""

from pydoc import text
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pdfplumber

from corpus.schemas import Section


class BaseParser(ABC):
    act: str = ""

    @abstractmethod
    def parse(self, pdf_path: Path) -> list[Section]:
        ...

    def preprocess(self, text: str) -> str:  
        """override in subclasses for act-specific preprocessing. default: no changes."""
        return text

    @staticmethod
    def extract_raw_text(pdf_path: Path) -> str:
        """pulls text out of every page, joined, with repeated headers/footers stripped."""
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        pages = BaseParser._strip_running_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _strip_running_headers(pages: list[str]) -> list[str]:
        """
        a line repeated across most pages (act name, page number) is
        boilerplate, not section text - drop it everywhere.
        """
        if len(pages) < 3:
            return pages

        line_counts: dict[str, int] = {}
        for page in pages:
            for line in page.splitlines():
                line = line.strip()
                if line:
                    line_counts[line] = line_counts.get(line, 0) + 1

        threshold = len(pages) * 0.5
        boilerplate = {line for line, count in line_counts.items() if count >= threshold}

        cleaned = []
        for page in pages:
            kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
            cleaned.append("\n".join(kept))
        return cleaned


# matches "302. Murder.—" / "304A. Causing death by negligence.—"
# group 1 = number, group 2 = title
SECTION_START = re.compile(
    r'(?:^|\n)\s*(\d{1,3}[A-Z]{0,2})\.\s+([A-Z][^.\n]*?)\.\s*[-—–]\s*',
    re.MULTILINE,
)

# matches a "CHAPTER XVI" line followed by its title on the next line
CHAPTER_START = re.compile(
    r'(?:^|\n)\s*CHAPTER\s+([IVXLC\d]+)\s*\n\s*([^\n]*)',
    re.MULTILINE,
)


class ChapterSectionParser(BaseParser):
    """
    shared implementation for acts with a flat Chapter -> Section grammar
    (IPC, BNS, BNSS, CPC). subclasses set `act`, `default_status`,
    `effective_date`, and optionally override `extra_metadata()` to attach
    act-specific extras (e.g. IPC's replaced_by mapping).
    """

    default_status: str = "active"
    effective_date: str = ""

    def parse(self, pdf_path: Path) -> list[Section]:
        raw_text = self.extract_raw_text(pdf_path)
        text = self.preprocess(raw_text)
        return self._split_into_sections(text)

    def extra_metadata(self, number: str) -> dict:
        """override in subclasses for act-specific extras. default: nothing extra."""
        return {}

    def _split_into_sections(self, raw_text: str) -> list[Section]:
        chapters = list(CHAPTER_START.finditer(raw_text))
        matches = list(SECTION_START.finditer(raw_text))
        sections = []

        for i, match in enumerate(matches):
            number = match.group(1)
            title = match.group(2).strip()

            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            body = raw_text[body_start:body_end].strip()

            if not body:
                continue  # false-positive match (e.g. a number inside a footnote)

            chapter = self._label_for_position(chapters, match.start())

            metadata = {"chapter": chapter, "effective_date": self.effective_date}
            metadata.update(self.extra_metadata(number))

            sections.append(Section(
                act=self.act,
                unit_type="section",
                number=number,
                title=title,
                body=body,
                status=self.default_status,
                metadata=metadata,
            ))

        return sections

    @staticmethod
    def _label_for_position(headers, position: int) -> str:
        """finds the chapter header that comes right before this position in the text."""
        current = ""
        for header_match in headers:
            if header_match.start() > position:
                break
            number, title = header_match.group(1), header_match.group(2).strip()
            current = f"Chapter {number}: {title}" if title else f"Chapter {number}"
        return current