from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from qdrant_client import QdrantClient, models

from agentic_rag.config import get_settings

TEXT_DENSE_VECTOR_NAME = "text-dense"
TEXT_SPARSE_VECTOR_NAME = "text-sparse"
IMAGE_VECTOR_NAME = "colpali"


class RagType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings().qdrant
    return QdrantClient(url=settings.url, api_key=settings.api_key)


def collection_name(kb_name: str, rag_type: RagType) -> str:
    return f"{kb_name}_{rag_type.value}"


def collection_exists(name: str) -> bool:
    return get_qdrant_client().collection_exists(collection_name=name)


def ensure_image_collection(kb_name: str) -> str:
    settings = get_settings().colpali
    client = get_qdrant_client()
    name = collection_name(kb_name=kb_name, rag_type=RagType.IMAGE)

    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                IMAGE_VECTOR_NAME: models.VectorParams(
                    size=settings.vector_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                )
            },
        )

    return name


def ensure_collection(kb_name: str, rag_type: RagType) -> str:
    if rag_type is RagType.IMAGE:
        return ensure_image_collection(kb_name=kb_name)
    return collection_name(kb_name=kb_name, rag_type=RagType.TEXT)


def delete_kb(kb_name: str) -> None:
    client = get_qdrant_client()

    for rag_type in RagType:
        name = collection_name(kb_name=kb_name, rag_type=rag_type)
        if client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)


def list_kbs() -> list[str]:
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}

    kb_names: set[str] = set()
    for rag_type in RagType:
        suffix = f"_{rag_type.value}"
        kb_names.update(name[: -len(suffix)] for name in names if name.endswith(suffix))

    return sorted(kb_names)
