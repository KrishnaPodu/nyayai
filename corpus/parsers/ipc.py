"""
IPC Parser – tailor-made for the Indian Penal Code PDF layout.
Exposes IPCParser with a parse(Path) -> list[Section] method.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber  # robust extraction for legal PDFs

from corpus.schemas import Section, Passage   # schema shared across all acts


class IPCParser:
    """
    Parses the IPC Act PDF into Section objects.
    Usage:
        parser = IPCParser()
        sections = parser.parse(Path("ipc.pdf"))
    """

    def parse(self, pdf_path: Path) -> List[Section]:
        """Public entry point – takes a PDF path, returns Sections."""
        raw_text = self._extract_text(pdf_path)
        text = self._load_act_text(raw_text)           # discard TOC
        text = self._strip_page_numbers(text)          # lone numbers at top of pages
        text = self._remove_editorial_marks(text)      # 1[, 2***, 3[...] etc.
        text = self._remove_footnotes(text)            # 1. Subs. by... etc.
        chapter_blocks = self._split_chapters(text)

        sections = []
        for ch in chapter_blocks:
            for sec in self._split_sections(ch['text']):
                sections.append(Section(
                    act="IPC",
                    unit_type="section",
                    number=sec['section_number'],
                    title=sec['title'],
                    body=sec['body'],
                    status="active",
                    metadata={
                        "chapter": ch['num'],
                        "chapter_title": ch['title'],
                        "effective_date": "1860-01-01",
                    }
                ))
        return sections

    # ------------------------------------------------------------------
    # Internal helpers (each does one job)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(pdf_path: Path) -> str:
        """Read all text from the IPC PDF, joining pages with newlines."""
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    @staticmethod
    def _load_act_text(text: str) -> str:
        """Skip the Arrangement of Sections (TOC)."""
        match = re.search(r'ACT\s+NO\.\s+45\s+OF\s+1860', text, re.IGNORECASE)
        if not match:
            raise ValueError(
                "Could not locate the start of the IPC Act. "
                "Expected 'ACT NO. 45 OF 1860'."
            )
        return text[match.start():]

    @staticmethod
    def _strip_page_numbers(text: str) -> str:
        """Remove lines that contain only a number (page numbers)."""
        return re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    @staticmethod
    def _remove_editorial_marks(text: str) -> str:
        """
        Delete inline amendment markers inserted by IndiaCode:
        1[some text], 2***, 3[...], 5**** etc.
        """
        text = re.sub(r'\d+\[[^\]]*\]', '', text)
        text = re.sub(r'\d+\*+', '', text)
        return text

    @staticmethod
    def _remove_footnotes(text: str) -> str:
        """
        Remove footnote blocks that start with '1. Subs.', '2. Ins.', etc.
        These are amendment notes and should never enter embeddings.
        """
        lines = text.splitlines(keepends=True)
        out_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            if re.match(r'^\d+\.\s+(Subs\.|Ins\.|Omitted|Added|Rep\.)\b', line):
                i += 1
                # Skip continuation lines (blank or indented)
                while i < n:
                    nxt = lines[i]
                    if nxt.strip() == '' or re.match(r'^\s+\S', nxt):
                        i += 1
                    else:
                        break
            else:
                out_lines.append(line)
                i += 1

        return ''.join(out_lines)

    @staticmethod
    def _split_chapters(text: str) -> List[dict]:
        """
        Split the text into chapter blocks using:
            CHAPTER <ROMAN>
            <TITLE>
        Returns a list of dicts with keys 'num', 'title', 'text'.
        """
        pattern = re.compile(
            r'^CHAPTER\s+([IVXLCDM]+)\s*\n\s*(.+)',
            re.MULTILINE
        )
        matches = list(pattern.finditer(text))

        chapter_blocks = []
        for idx, m in enumerate(matches):
            chapter_num = m.group(1)
            chapter_title = m.group(2).strip()
            body_start = m.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chapter_text = text[body_start:body_end]

            chapter_blocks.append({
                'num': chapter_num,
                'title': chapter_title,
                'text': chapter_text
            })

        return chapter_blocks

    @staticmethod
    def _split_sections(chapter_text: str) -> List[dict]:
        """
        Split a chapter's text into individual sections.
        A section header is:
            <number>[optional letter]. Title.—
        """
        sec_pattern = re.compile(
            r'^(\d+[A-Z]?)\.\s+(.*?)\.\s*(?:—|--)',   # em dash or two hyphens
            re.MULTILINE
        )
        matches = list(sec_pattern.finditer(chapter_text))

        sections = []
        for idx, m in enumerate(matches):
            section_number = m.group(1)
            section_title = m.group(2).strip()
            body_start = m.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(chapter_text)
            body = chapter_text[body_start:body_end].strip()

            sections.append({
                'section_number': section_number,
                'title': section_title,
                'body': body
            })

        return sections