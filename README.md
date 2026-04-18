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
