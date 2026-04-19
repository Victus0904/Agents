"""FastAPI service wrapper for the multimodal RAG pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from multimodal_rag import Settings, ask_question, ingest_pdf


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Multimodal RAG API",
    version="1.0.0",
    description="API for PDF ingestion and question answering with Gemini and ChromaDB.",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    pdf: UploadFile = File(...),
    write_debug_json: bool = Form(False),
) -> dict:
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    destination = UPLOAD_DIR / Path(pdf.filename).name
    with destination.open("wb") as file_handle:
        shutil.copyfileobj(pdf.file, file_handle)

    try:
        settings = Settings.from_env()
        return ingest_pdf(str(destination), settings, write_debug_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        pdf.file.close()


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    try:
        settings = Settings.from_env()
        return ask_question(payload.question, settings, payload.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
