"""Google Gemini + ChromaDB multimodal RAG pipeline."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types

from document_parser import parse_pdf, save_processed_metadata


SUPPORTED_TEXT_TYPES = {
    "CompositeElement",
    "NarrativeText",
    "Title",
    "ListItem",
    "Text",
}


@dataclass(slots=True)
class Settings:
    gemini_api_key: str
    chroma_path: str = "chroma_db"
    chroma_collection: str = "document_chunks"
    vision_model: str = "gemini-2.5-flash"
    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 50
    api_retry_attempts: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")

        return cls(
            gemini_api_key=gemini_api_key,
            chroma_path=os.getenv("CHROMA_PATH", "chroma_db"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "document_chunks"),
            vision_model=os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"),
            generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
            api_retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS", "5")),
        )


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    @staticmethod
    def _clean_text(value: str) -> str:
        return value.strip()

    @staticmethod
    def _require_non_empty(value: str, context: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"Cannot process empty content for {context}.")
        return cleaned

    @staticmethod
    def _retry_delay_seconds(exc: Exception) -> float | None:
        message = str(exc)
        retry_in_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
        if retry_in_match:
            return float(retry_in_match.group(1))

        retry_delay_match = re.search(r"'retryDelay': '([0-9]+)s'", message)
        if retry_delay_match:
            return float(retry_delay_match.group(1))

        return None

    def _run_with_retry(self, operation: str, func: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.api_retry_attempts + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc)
                if "RESOURCE_EXHAUSTED" not in message and "quota" not in message.lower():
                    raise

                if attempt >= self.settings.api_retry_attempts:
                    break

                delay = self._retry_delay_seconds(exc) or min(60.0, float(attempt * 10))
                print(
                    f"{operation} hit Gemini quota limits. "
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{self.settings.api_retry_attempts})."
                )
                time.sleep(delay)

        raise RuntimeError(
            f"{operation} failed after {self.settings.api_retry_attempts} attempts: {last_error}"
        ) from last_error

    def _generate_text(self, model: str, contents: Any, context: str) -> str:
        response = self._run_with_retry(
            operation=context,
            func=lambda: self.client.models.generate_content(model=model, contents=contents),
        )
        text = self._clean_text(response.text or "")
        if not text:
            raise ValueError(f"Gemini returned empty text for {context}.")
        return text

    def summarize_table(self, table_html: str, page_number: int | None) -> str:
        table_html = self._require_non_empty(table_html, "table summarization")
        prompt = (
            "You are preparing table text for multimodal retrieval.\n"
            "Summarize the following table extracted from a PDF.\n"
            "Return concise factual text that preserves headers, row/column relationships, "
            "key metrics, trends, units, dates, and comparisons.\n"
            "Do not mention HTML or speculate beyond the table.\n\n"
            f"Page: {page_number}\n"
            f"Table HTML:\n{table_html}"
        )
        return self._generate_text(
            model=self.settings.vision_model,
            contents=prompt,
            context="table summarization",
        )

    def summarize_image(self, image_path: str, page_number: int | None) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image asset not found: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "image/jpeg"
        image_bytes = path.read_bytes()

        prompt = (
            "You are preparing image text for multimodal retrieval.\n"
            "Summarize this extracted PDF image in concise factual prose.\n"
            "Describe charts, figures, labels, legends, visible numbers, axes, categories, and "
            "other visual evidence that could answer a later user query.\n"
            "If text is visible in the image, include it when important.\n"
            f"Page: {page_number}"
        )
        return self._generate_text(
            model=self.settings.vision_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            context="image summarization",
        )

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [self._require_non_empty(text, "embedding") for text in texts]
        all_embeddings: list[list[float]] = []

        for start in range(0, len(cleaned_texts), self.settings.embedding_batch_size):
            batch = cleaned_texts[start : start + self.settings.embedding_batch_size]
            result = self._run_with_retry(
                operation="embedding",
                func=lambda batch=batch: self.client.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=self.settings.embedding_dimensions,
                    ),
                ),
            )
            if not result.embeddings:
                raise ValueError("Gemini returned no embeddings.")

            all_embeddings.extend([list(item.values) for item in result.embeddings])

        if len(all_embeddings) != len(cleaned_texts):
            raise ValueError("Gemini returned an unexpected number of embeddings.")

        return all_embeddings

    def answer_question(
        self,
        question: str,
        retrieved_rows: list[dict[str, Any]],
    ) -> str:
        if not retrieved_rows:
            return "No relevant context was retrieved from the vector store for this question."

        context_lines: list[str] = []
        image_parts: list[Any] = []
        seen_asset_paths: set[str] = set()

        for row in retrieved_rows:
            context_lines.append(
                "\n".join(
                    [
                        f"Element ID: {row['element_id']}",
                        f"Chunk type: {row['chunk_type']}",
                        f"Source: {row['source_path']}",
                        f"Page: {row['page_number']}",
                        f"Distance: {row['cosine_distance']}",
                        f"Summary: {row['summary']}",
                        f"Original content: {row['content']}",
                    ]
                )
            )

            asset_path = row.get("asset_path")
            if asset_path and asset_path not in seen_asset_paths:
                path = Path(asset_path)
                if path.exists():
                    mime_type, _ = mimetypes.guess_type(path.name)
                    image_parts.append(
                        types.Part.from_bytes(
                            data=path.read_bytes(),
                            mime_type=mime_type or "image/jpeg",
                        )
                    )
                    seen_asset_paths.add(asset_path)

        context_block = "\n\n".join(context_lines)
        prompt = (
            "Answer the user's question using only the retrieved multimodal evidence.\n"
            "Use the text summaries and original content below, and inspect any attached original "
            "images when relevant.\n"
            "If the evidence is incomplete or ambiguous, say so explicitly.\n"
            "Cite the supporting page numbers in your answer when possible.\n\n"
            f"User question:\n{question}\n\n"
            "Retrieved context:\n"
            f"{context_block}"
        )

        return self._generate_text(
            model=self.settings.generation_model,
            contents=[prompt, *image_parts],
            context="final answer generation",
        )


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_document(
        self,
        source_path: str,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            raise ValueError(f"No indexable records were produced for document: {source_path}")

        existing = self.collection.get(where={"source_path": source_path}, include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        ids = [record["element_id"] for record in records]
        embeddings = [record["embedding"] for record in records]
        documents = [record["summary"] for record in records]
        metadatas = [
            {
                "element_id": record["element_id"],
                "source_path": record["source_path"],
                "chunk_type": record["chunk_type"],
                "page_number": record["page_number"] if record["page_number"] is not None else -1,
                "content": record["content"],
                "summary": record["summary"],
                "asset_path": record.get("asset_path") or "",
            }
            for record in records
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for index, element_id in enumerate(ids):
            metadata = metadatas[index]
            page_number = metadata.get("page_number", -1)
            rows.append(
                {
                    "id": element_id,
                    "element_id": metadata.get("element_id", element_id),
                    "source_path": metadata.get("source_path"),
                    "chunk_type": metadata.get("chunk_type"),
                    "page_number": None if page_number == -1 else page_number,
                    "content": metadata.get("content", ""),
                    "summary": documents[index],
                    "asset_path": metadata.get("asset_path") or None,
                    "metadata": metadata,
                    "cosine_distance": distances[index],
                }
            )

        return rows


def build_records(
    elements: list[dict[str, Any]],
    gemini_client: GeminiClient,
) -> list[dict[str, Any]]:
    records_without_embeddings: list[dict[str, Any]] = []

    for element in elements:
        element_id = element["element_id"]
        chunk_type = element["type"]
        metadata = element["metadata"]
        page_number = metadata.get("page_number")
        source_path = metadata["source"]
        asset_path = metadata.get("image_path")
        content = element.get("text_content", "").strip()

        try:
            if chunk_type == "Table":
                if not content:
                    continue
                summary = gemini_client.summarize_table(content, page_number)
            elif chunk_type == "Image" and asset_path:
                # The image summary is what gets embedded for retrieval, while the
                # original local file path is preserved in metadata so the exact
                # extracted JPG/PNG can be loaded later for final generation.
                summary = gemini_client.summarize_image(asset_path, page_number)
            elif chunk_type in SUPPORTED_TEXT_TYPES and content:
                summary = content
            else:
                continue

            if not summary:
                continue

            records_without_embeddings.append(
                {
                    "element_id": element_id,
                    "source_path": source_path,
                    "chunk_type": chunk_type,
                    "page_number": page_number,
                    "content": content,
                    "summary": summary,
                    "asset_path": asset_path,
                    "metadata": metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Skipping element {element_id} ({chunk_type}) due to processing error: {exc}"
            )

    if not records_without_embeddings:
        return []

    embeddings = gemini_client.embed_texts([record["summary"] for record in records_without_embeddings])
    for record, embedding in zip(records_without_embeddings, embeddings, strict=True):
        record["embedding"] = embedding

    return records_without_embeddings


def ingest_pdf(
    pdf_path: str,
    settings: Settings,
    write_debug_json: bool,
) -> dict[str, Any]:
    gemini_client = GeminiClient(settings)
    vector_store = VectorStore(settings)

    elements = parse_pdf(pdf_path)
    if write_debug_json:
        save_processed_metadata(elements)

    records = build_records(elements, gemini_client)
    source_path = str(Path(pdf_path).expanduser().resolve())
    vector_store.replace_document(source_path, records)

    return {
        "source_path": source_path,
        "records_indexed": len(records),
        "chroma_path": settings.chroma_path,
        "collection": settings.chroma_collection,
    }


def ask_question(question: str, settings: Settings, top_k: int) -> dict[str, Any]:
    gemini_client = GeminiClient(settings)
    vector_store = VectorStore(settings)

    query_embedding = gemini_client.embed_text(question)
    retrieved_rows = vector_store.search(query_embedding, top_k)
    answer = gemini_client.answer_question(question, retrieved_rows)

    response = {
        "question": question,
        "matches": [
            {
                "id": row["id"],
                "element_id": row["element_id"],
                "chunk_type": row["chunk_type"],
                "source_path": row["source_path"],
                "page_number": row["page_number"],
                "asset_path": row["asset_path"],
                "cosine_distance": row["cosine_distance"],
                "summary": row["summary"],
            }
            for row in retrieved_rows
        ],
        "answer": answer,
    }
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal PDF RAG with Unstructured, Gemini, and ChromaDB."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse a PDF and index it.")
    ingest_parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    ingest_parser.add_argument(
        "--write-debug-json",
        action="store_true",
        help="Also write processed_metadata.json from the parser output.",
    )

    ask_parser = subparsers.add_parser("ask", help="Query the indexed documents.")
    ask_parser.add_argument("--question", required=True, help="User question.")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "ingest":
        print(json.dumps(ingest_pdf(args.pdf, settings, args.write_debug_json), indent=2))
    elif args.command == "ask":
        print(json.dumps(ask_question(args.question, settings, args.top_k), indent=2))


if __name__ == "__main__":
    main()
