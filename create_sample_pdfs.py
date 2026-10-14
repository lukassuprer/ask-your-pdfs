"""Create the three small, non-sensitive sample PDFs used by the README."""

from pathlib import Path

import fitz


DOCUMENTS = {
    "rag-overview.pdf": (
        "RAG Overview\n\n"
        "Retrieval-augmented generation combines a search step with language-model "
        "generation. A typical pipeline ingests source documents, extracts text, "
        "splits it into overlapping chunks, creates an embedding for each chunk, "
        "stores those vectors, retrieves the most relevant chunks for a question, "
        "and gives the retrieved context to a language model. Retrieval grounds "
        "the answer in the source corpus and can reduce unsupported claims."
    ),
    "vector-search-guide.pdf": (
        "Vector Search Guide\n\n"
        "An embedding is a numeric representation of text in which semantically "
        "similar texts tend to be close together. Cosine similarity measures the "
        "angle between two vectors; a larger cosine similarity generally means "
        "the texts are more semantically alike. A vector database indexes "
        "embeddings and returns nearest neighbors efficiently. Metadata filters "
        "can restrict results to a file, page range, or other attribute."
    ),
    "python-project-guide.pdf": (
        "Python Project Guide\n\n"
        "A virtual environment isolates a project's Python packages from the "
        "system interpreter and from other projects. This improves reproducibility "
        "because a requirements file can describe the dependencies installed in "
        "that environment. A practical workflow is to create .venv, activate it, "
        "install dependencies, and run tests before sharing the project."
    ),
}


def main() -> None:
    output_dir = Path("documents")
    output_dir.mkdir(exist_ok=True)
    for filename, text in DOCUMENTS.items():
        document = fitz.open()
        page = document.new_page()
        page.insert_textbox((72, 72, 540, 700), text, fontsize=12, lineheight=1.5)
        document.save(output_dir / filename)
        document.close()
    print(f"Created {len(DOCUMENTS)} PDFs in {output_dir}/")


if __name__ == "__main__":
    main()
