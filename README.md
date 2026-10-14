# Ask-Your-PDFs (RAG)

Small Python retrieval-augmented generation (RAG) application for asking questions about a folder of PDFs. It extracts page text, creates overlapping chunks, embeds them with `sentence-transformers`, stores them in persistent Chroma, retrieves the most relevant chunks, and asks Ollama Cloud to answer with page-level citations.

## Setup

```bash
cd ask-your-pdfs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Put your Ollama Cloud API key in .env, or export OLLAMA_API_KEY in your shell.
```

The default embedding model is `all-MiniLM-L6-v2`. The first run downloads it from Hugging Face and caches it locally.

Create an Ollama Cloud API key at https://ollama.com/settings/keys. The default generation model is `gemma4:31b`.

## Ingest PDFs

Put PDFs in `documents/`, then run:

```bash
python ingest.py
```

Useful options:

```bash
python ingest.py --pdf-dir path/to/pdfs --db-dir chroma_db --chunk-size 900 --overlap 150
```

Each stored chunk has `filename` and one-based `page` metadata. Re-running ingestion safely upserts the same chunk IDs.

## Ask a question

```bash
python query.py "What is retrieval-augmented generation?"
```

Without a command-line question, the script prompts interactively:

```bash
python query.py
```

The answer is constrained to retrieved excerpts and should cite claims like `[rag-overview.pdf, p. 1]`. If the corpus does not support an answer, it responds: `I don't know based on the provided documents.`

## Sample questions and expected citations

The repository includes three short, non-sensitive sample study guides in `documents/`:

| Question | Expected citation |
| --- | --- |
| What are the main stages of a RAG pipeline? | `[rag-overview.pdf, p. 1]` |
| What does cosine similarity measure? | `[vector-search-guide.pdf, p. 1]` |
| Why should Python projects use virtual environments? | `[python-project-guide.pdf, p. 1]` |

These expected citations identify the document/page containing the answer; exact wording can vary with the selected LLM.

## Project structure

* `ingest.py` — PDF extraction, chunking, embedding, and Chroma persistence.
* `query.py` — CLI retrieval and LLM answer generation.
* `src/rag.py` — reusable pipeline implementation.
* `documents/` — sample PDFs.
* `tests/` — tests for chunk overlap and page metadata.

## Design notes

The implementation uses PyMuPDF for extraction, word-based chunks of 900 words with 150 words of overlap, normalized MiniLM embeddings, cosine-distance Chroma search, and Ollama Cloud's `/api/chat` endpoint for generation. The LLM receives the retrieved text together with its source label, and the system prompt explicitly forbids unsupported answers and invented citations.
