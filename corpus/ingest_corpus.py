"""
thin CLI wrapper around corpus/ingest.py.

usage:
    uv run python scripts/ingest_corpus.py --act ipc
    uv run python scripts/ingest_corpus.py --act bns --force
    uv run python scripts/ingest_corpus.py --all
"""

import argparse

from corpus.ingest import ingest_act
from corpus.embeddings import PassageEmbedder
from config.constants import ACTS

ALL_ACTS = ACTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", choices=ALL_ACTS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.act and not args.all:
        parser.error("pass --act <name> or --all")

    acts = ALL_ACTS if args.all else [args.act]

    # load InLegalBERT once, reuse across acts - it's the slow part of the pipeline
    embedder = PassageEmbedder()

    for act in acts:
        ingest_act(act, force=args.force, embedder=embedder)


if __name__ == "__main__":
    main()