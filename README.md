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

## API Deployment

The repository now exposes the RAG pipeline as a FastAPI service for local
development and Render deployment.

Start it with Docker Compose:

```bash
docker compose up --build
```

On PowerShell:

```powershell
docker compose up --build
```

The API is available at `http://localhost:8000` with:

- `GET /health`
- `POST /ingest`
- `POST /ask`

OpenAPI docs are served at `http://localhost:8000/docs`.

### Ingest a PDF over HTTP

Using `curl`:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "pdf=@Optical_Music_Recognition_State_of_the_Art_and_Maj.pdf" \
  -F "write_debug_json=true"
```

Using PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/ingest" `
  -F "pdf=@Optical_Music_Recognition_State_of_the_Art_and_Maj.pdf" `
  -F "write_debug_json=true"
```

### Ask a question over HTTP

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this paper about?","top_k":5}'
```

On PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"What is this paper about?\",\"top_k\":5}'
```

### Render

This repo includes `render.yaml` for a Docker-based Render web service. Render
free is suitable for demos, but there is an important storage constraint:
`ChromaDB` in this repo uses local disk, while Render free web services do not
provide persistent disks. That means indexed data may be lost on restart,
redeploy, or spin-down.

For a demo deployment on Render:

1. Push this repo to GitHub.
2. Create a new Render Blueprint or Web Service from the repo.
3. Set the required environment variables from `.env.example`.
4. Deploy the service and use the `/docs` page or the API endpoints directly.

Notes:

- The image installs `poppler-utils`, `tesseract-ocr`, `libmagic-dev`, and related runtime libraries.
- `./chroma_db` is mounted locally in Compose so your index persists across local restarts.
- `./uploads` is mounted locally in Compose, so uploaded PDFs and extracted assets persist across local restarts.
- Render free remains demo-grade unless you move vector persistence off local disk or upgrade to a plan with persistent storage.

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
- `multimodal_api.py`: FastAPI service for ingestion and question answering
- `multimodal_rag.py`: ingestion, retrieval, and generation CLI/core logic
- `render.yaml`: Render web service definition
- `.env.example`: required environment variables

## Current behavior notes

- Text chunks are embedded directly from extracted text.
- Table chunks are summarized into retrieval-oriented text before embedding.
- Image chunks are summarized with Gemini Vision, embedded as text, and keep the local
  extracted file path in metadata for final multimodal answer generation.
- Re-ingesting the same PDF replaces previously indexed chunks for that source path.
