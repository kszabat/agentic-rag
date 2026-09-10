from __future__ import annotations

import uuid
from pathlib import Path

import pypdfium2 as pdfium
from qdrant_client import models

from agentic_rag.config import get_settings
from agentic_rag.embeddings.colpali_embedding import colpali_embed_documents
from agentic_rag.vector_store.qdrant_manager import (
    IMAGE_VECTOR_NAME,
    RagType,
    ensure_image_collection,
    get_qdrant_client,
)


def render_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    settings = get_settings().colpali
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = settings.pdf_render_dpi / 72  # pypdfium2 scale=1 equals 72 dpi

    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)

    image_paths: list[Path] = []
    try:
        for page_index in range(n_pages):
            page = pdf[page_index]
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                image_output_path = output_dir / f"page_{page_index:04d}.png"
                image.save(image_output_path)
                image_paths.append(image_output_path)
            finally:
                page.close()
    finally:
        pdf.close()

    return image_paths


def ingest_image_document(kb_name: str, file_path: Path) -> int:
    data_dir = get_settings().data_dir
    doc_name = file_path.stem
    images_dir = data_dir / "images" / "kb_name" / doc_name

    image_paths = render_pdf_to_images(pdf_path=file_path, output_dir=images_dir)
    if not image_paths:
        return 0

    embeddings = colpali_embed_documents(image_paths=image_paths)
    collection = ensure_image_collection(kb_name=kb_name)

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={IMAGE_VECTOR_NAME: embedding},
            payload={
                "kb_name": kb_name,
                "doc_id": doc_name,
                "source_file": str(file_path),
                "image_path": str(image_path),
                "page_number": page_index,
            },
        )
        for page_index, (image_path, embedding) in enumerate(
            zip(image_paths, embeddings)
        )
    ]

    get_qdrant_client().upsert(collection_name=collection, points=points)
    return len(points)
