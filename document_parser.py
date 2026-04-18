"""Layout-aware PDF parsing for a multimodal RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf


def _coerce_text(element: Any, element_type: str) -> str:
    metadata = getattr(element, "metadata", None)
    if element_type == "Table":
        return getattr(metadata, "text_as_html", None) or getattr(element, "text", "") or ""
    return getattr(element, "text", "") or ""


def _resolve_image_path(metadata: Any, assets_dir: Path) -> str | None:
    candidates = [
        getattr(metadata, "image_path", None),
        getattr(metadata, "filename", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() and candidate_path.exists():
            return str(candidate_path)
        joined_path = assets_dir / candidate_path.name
        if joined_path.exists():
            return str(joined_path.resolve())
    return None


def parse_pdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Parse a PDF into normalized text, table, and image elements."""
    source_path = Path(pdf_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF file not found: {source_path}")

    assets_dir = source_path.parent / "extracted_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(source_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(assets_dir),
    )

    processed_elements: list[dict[str, Any]] = []
    for index, element in enumerate(elements):
        element_type = type(element).__name__
        metadata = getattr(element, "metadata", None)
        image_path = _resolve_image_path(metadata, assets_dir) if metadata else None
        page_number = getattr(metadata, "page_number", None)
        text_content = _coerce_text(element, element_type).strip()
        element_id = f"{source_path.name}:{page_number or 'na'}:{index}:{element_type}"

        processed_elements.append(
            {
                "element_id": element_id,
                "type": element_type,
                "text_content": text_content,
                "metadata": {
                    "page_number": page_number,
                    "source": str(source_path),
                    "image_path": image_path,
                },
            }
        )

    return processed_elements


def save_processed_metadata(
    processed_elements: list[dict[str, Any]],
    output_path: str | Path = "processed_metadata.json",
) -> Path:
    """Save parsed metadata to disk for inspection or debugging."""
    destination = Path(output_path).expanduser().resolve()
    destination.write_text(
        json.dumps(processed_elements, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    sample_pdf_path = "sample_report.pdf"
    try:
        processed_elements = parse_pdf(sample_pdf_path)
        output_path = save_processed_metadata(processed_elements)
        print(f"Processed metadata saved to: {output_path}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error while parsing PDF: {exc}")


if __name__ == "__main__":
    main()
