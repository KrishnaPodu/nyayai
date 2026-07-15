"""
top-level ingestion for one act: parse -> chunk -> embed -> upload.
called by scripts/ingest_corpus.py.
"""

from pathlib import Path

from surya import settings

from corpus.parser import parse_act
from corpus.chunker import chunk_sections
from corpus.embeddings import PassageEmbedder
from corpus.uploader import get_client, ensure_collection, drop_act, upload_passages
from config.settings import settings

SOURCE_DIR = settings.corpus_sources_dir  # Path to the directory containing act PDFs


def ingest_act(act: str, force: bool = False, embedder: PassageEmbedder | None = None) -> int:
    pdf_path = _find_pdf(act)
    sections = parse_act(pdf_path, act)
    passages = chunk_sections(sections)

    embedder = embedder or PassageEmbedder()
    vectors = embedder.embed_passages(passages)

    client = get_client()
    ensure_collection(client)

    if force:
        drop_act(client, act)

    count = upload_passages(client, passages, vectors)
    print(f"{act}: {len(sections)} sections -> {len(passages)} passages -> {count} points uploaded")
    return count


def _find_pdf(act: str) -> Path:
    act_dir = SOURCE_DIR / act.lower()
    pdfs = list(act_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF found in {act_dir}")
    if len(pdfs) > 1:
        print(f"warning: multiple PDFs in {act_dir}, using {pdfs[0].name}")
    return pdfs[0]