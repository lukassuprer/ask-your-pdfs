#!/usr/bin/env python3
"""Ingest PDFs into the persistent Chroma vector database."""

import argparse
from pathlib import Path

from src.rag import DEFAULT_EMBEDDING_MODEL, ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("documents"))
    parser.add_argument("--db-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    count = ingest(
        args.pdf_dir,
        args.db_dir,
        args.embedding_model,
        args.chunk_size,
        args.overlap,
    )
    print(f"Ingested {count} chunks from {args.pdf_dir} into {args.db_dir}.")


if __name__ == "__main__":
    main()
