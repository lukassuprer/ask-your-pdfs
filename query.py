#!/usr/bin/env python3
"""Ask questions about the ingested PDFs."""

import argparse
from pathlib import Path

from src.rag import DEFAULT_EMBEDDING_MODEL, generate_answer, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--db-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    question = args.question or input("Question: ")
    chunks = retrieve(question, args.db_dir, args.embedding_model, args.top_k)
    print(generate_answer(question, chunks))


if __name__ == "__main__":
    main()
