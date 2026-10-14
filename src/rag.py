"""Core ingestion, retrieval, and answer-generation components."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
load_dotenv()


@dataclass(frozen=True)
class Chunk:
    text: str
    filename: str
    page: int
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"[{self.filename}, p. {self.page}]"


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into word-based chunks while preserving a useful overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")

    words = re.findall(r"\S+", text)
    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def extract_chunks(pdf_dir: Path, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Extract page text from every PDF and split it into citation-preserving chunks."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    chunks: list[Chunk] = []
    for pdf_path in pdfs:
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                for chunk_index, text_chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                    chunks.append(
                        Chunk(
                            text=text_chunk,
                            filename=pdf_path.name,
                            page=page_index,
                            chunk_id=f"{pdf_path.name}:{page_index}:{chunk_index}",
                        )
                    )
    if not chunks:
        raise ValueError(f"PDF files in {pdf_dir} contained no extractable text")
    return chunks


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()


def get_collection(db_dir: Path, collection_name: str = DEFAULT_COLLECTION):
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(
    pdf_dir: Path,
    db_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 900,
    overlap: int = 150,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    chunks = extract_chunks(pdf_dir, chunk_size, overlap)
    embedder = Embedder(model_name)
    collection = get_collection(db_dir, collection_name)
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embedder.encode([chunk.text for chunk in chunks]),
        metadatas=[
            {"filename": chunk.filename, "page": chunk.page}
            for chunk in chunks
        ],
    )
    return len(chunks)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    page: int
    distance: float

    @property
    def citation(self) -> str:
        return f"[{self.filename}, p. {self.page}]"


def retrieve(
    question: str,
    db_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[RetrievedChunk]:
    if not question.strip():
        raise ValueError("Question must not be empty")
    collection = get_collection(db_dir, collection_name)
    if collection.count() == 0:
        raise ValueError("The vector database is empty. Run ingest.py first.")

    query_embedding = Embedder(model_name).encode([question])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(
            text=document,
            filename=metadata["filename"],
            page=int(metadata["page"]),
            distance=float(distance),
        )
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def build_context(chunks: Iterable[RetrievedChunk]) -> str:
    return "\n\n".join(f"{chunk.citation}\n{chunk.text}" for chunk in chunks)


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    model: str | None = None,
) -> str:
    """Answer using an OpenAI-compatible API and require page-level citations."""
    if not chunks:
        return "I don't know based on the provided documents."

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied excerpts. If the excerpts do not "
                    "support an answer, say exactly: I don't know based on the provided "
                    "documents. Cite every factual claim with one or more citations in "
                    "the exact form [filename, p. N]. Do not invent citations."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nExcerpts:\n{build_context(chunks)}",
            },
        ],
    )
    return response.choices[0].message.content or "I don't know based on the provided documents."
