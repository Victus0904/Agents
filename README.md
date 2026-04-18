# Document Parser

This repository contains a Python utility, `document_parser.py`, for layout-aware
PDF partitioning with the `unstructured` library.

## What it does

- Extracts text, tables, and images from a local PDF
- Saves extracted image assets into `./extracted_assets/`
- Serializes normalized element metadata into `processed_metadata.json`
- Preserves tables as HTML when available for improved LLM consumption

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python document_parser.py
```

Update the placeholder `sample_report.pdf` path in `document_parser.py` or import
`parse_pdf()` into your own pipeline code.
