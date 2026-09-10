from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sentence_transformers import MultiVectorEncoder

from agentic_rag.config import get_settings


@lru_cache
def _get_model() -> MultiVectorEncoder:
    settings = get_settings().colpali
    return MultiVectorEncoder(settings.model_name, device=settings.device)


def colpali_embed_documents(image_paths: list[Path]) -> list[list[list[float]]]:
    document_embeddings = _get_model().encode_document([str(p) for p in image_paths])
    return [vec.tolist() for vec in document_embeddings]


def colpali_embed_query(query: str) -> list[list[float]]:
    embeddings = _get_model().encode_query([query])
    return embeddings[0].tolist()
