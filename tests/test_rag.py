from pathlib import Path

import pytest

from src.rag import chunk_text, extract_chunks


def test_chunk_text_has_overlap():
    chunks = chunk_text(" ".join(f"word{i}" for i in range(20)), chunk_size=8, overlap=2)
    assert len(chunks) == 3
    assert "word6" in chunks[0]
    assert "word6" in chunks[1]


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=4, overlap=4)


def test_extract_chunks_preserves_page_metadata(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "example.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A test page about retrieval.")
    document.save(pdf_path)
    document.close()

    chunks = extract_chunks(tmp_path, chunk_size=20, overlap=3)
    assert chunks[0].filename == "example.pdf"
    assert chunks[0].page == 1
