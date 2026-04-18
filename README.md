# Multimodal PDF RAG with Gemini and ChromaDB

This repository implements a multimodal RAG pipeline over PDF documents using:

- `unstructured` for layout-aware PDF partitioning
- Google Gemini API for table/image summarization, embeddings, and final answer generation
- embedded local `ChromaDB` for persistent vector storage and retrieval

## Pipeline

1. Ingestion: PDF -> text, tables, images
2. Summarization: tables and images -> Gemini summaries
3. Embedding: summaries -> Gemini embeddings
4. Storage: embeddings + metadata -> ChromaDB
5. Retrieval: question -> nearest vector matches
6. Generation: question + text context + original images -> Gemini answer

For image chunks, the embedding is generated from the Gemini text summary, while the
original extracted image path is stored in metadata so the exact asset can be loaded
during final answer generation.

## Requirements

- Python 3.10+
- A Google AI Studio API key for Gemini

Official references used for this implementation:

- Google GenAI SDK: https://ai.google.dev/gemini-api/docs/libraries
- Gemini image understanding: https://ai.google.dev/gemini-api/docs/image-understanding
- Gemini embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- ChromaDB docs: https://docs.trychroma.com/

## Installation

```bash
pip install -r requirements.txt
```

For `unstructured` PDF parsing, install the required system dependencies as well.
The upstream Unstructured installation guidance calls out `poppler-utils` and
`tesseract-ocr` for PDFs and images, plus `libmagic-dev` for file-type
detection. Docker setup below includes those packages.

Set environment variables, for example with `.env.example` as a template:

```bash
export GEMINI_API_KEY=...
export CHROMA_PATH=chroma_db
export CHROMA_COLLECTION=document_chunks
```

On PowerShell:

```powershell
$env:GEMINI_API_KEY="..."
$env:CHROMA_PATH="chroma_db"
$env:CHROMA_COLLECTION="document_chunks"
```

If you use a `.env` file for Docker or local tooling, copy `.env.example` and
replace the placeholder API key value with your real key.

## Ingest a PDF

```bash
python multimodal_rag.py ingest --pdf sample_report.pdf --write-debug-json
```

This will:

- extract text, tables, and images from the PDF
- write extracted images into `./extracted_assets/`
- optionally write `processed_metadata.json`
- summarize tables and images with Gemini
- embed the resulting summary text
- store vectors and metadata in persistent local ChromaDB storage

Chroma data is stored by default in `./chroma_db`.

## Docker Deployment

Build the image:

```bash
docker build -t multimodal-rag .
```

Create a `.env` file from `.env.example`, then run ingestion with the current
project directory mounted into the container:

```bash
docker run --rm \
  --env-file .env \
  -e CHROMA_PATH=/app/chroma_db \
  -v "$(pwd)/chroma_db:/app/chroma_db" \
  -v "$(pwd):/app/data" \
  -w /app/data \
  multimodal-rag ingest --pdf Optical_Music_Recognition_State_of_the_Art_and_Maj.pdf --write-debug-json
```

Ask a question against the persisted Chroma data:

```bash
docker run --rm \
  --env-file .env \
  -e CHROMA_PATH=/app/chroma_db \
  -v "$(pwd)/chroma_db:/app/chroma_db" \
  -v "$(pwd):/app/data" \
  -w /app/data \
  multimodal-rag ask --question "What is this paper about?" --top-k 5
```

If you prefer Docker Compose:

```bash
docker compose run --rm multimodal-rag ingest --pdf Optical_Music_Recognition_State_of_the_Art_and_Maj.pdf --write-debug-json
docker compose run --rm multimodal-rag ask --question "What is this paper about?" --top-k 5
```

Notes:

- The image installs `poppler-utils`, `tesseract-ocr`, and `libmagic-dev`.
- `./chroma_db` is mounted for persistent vector storage.
- The project directory is mounted at `/app/data`, so PDFs, extracted assets,
  and `processed_metadata.json` stay on the host filesystem.
- `CHROMA_PATH` is redirected to `/app/chroma_db` inside the container so the
  database lives on the dedicated volume mount.

## Ask questions

```bash
python multimodal_rag.py ask --question "What does the revenue chart show?" --top-k 5
```

The query flow:

- embeds the question
- retrieves the closest summaries from ChromaDB
- loads any original image files referenced by the retrieved rows
- sends the question, retrieved text context, and original images to Gemini

## Files

- `document_parser.py`: layout-aware PDF partitioning and debug JSON output
- `multimodal_rag.py`: ingestion, retrieval, and generation CLI
- `.env.example`: required environment variables

## Current behavior notes

- Text chunks are embedded directly from extracted text.
- Table chunks are summarized into retrieval-oriented text before embedding.
- Image chunks are summarized with Gemini Vision, embedded as text, and keep the local
  extracted file path in metadata for final multimodal answer generation.
- Re-ingesting the same PDF replaces previously indexed chunks for that source path.
