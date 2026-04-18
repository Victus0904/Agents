"""Layout-aware PDF parsing with Unstructured.

This script partitions a PDF into text, table, and image-aware elements and
serializes the extracted content into a JSON file for downstream RAG pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf


def parse_pdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Parse a PDF and return a normalized metadata payload."""
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

    for element in elements:
        element_type = type(element).__name__
        element_metadata = getattr(element, "metadata", None)

        if element_type == "Table":
            text_content = getattr(element_metadata, "text_as_html", None) or getattr(
                element, "text", ""
            )
        else:
            text_content = getattr(element, "text", "") or ""

        image_path = None
        if element_type == "Image" and element_metadata is not None:
            # Unstructured versions expose image output metadata differently.
            # When this element represents an extracted image, inspect the
            # metadata fields below to link the record to the saved JPG/PNG file
            # written into ./extracted_assets/.
            image_path = (
                getattr(element_metadata, "image_path", None)
                or getattr(element_metadata, "filename", None)
                or getattr(element_metadata, "image_base64", None)
            )

        processed_elements.append(
            {
                "type": element_type,
                "text_content": text_content,
                "metadata": {
                    "page_number": getattr(element_metadata, "page_number", None),
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
    """Persist extracted metadata to JSON."""
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
