from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from FlagEmbedding import BGEM3FlagModel
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.qdrant.utils import BatchSparseEncoding

from agentic_rag.config import get_settings

if TYPE_CHECKING:
    from collections import defaultdict

    import numpy as np

type SparseVector = tuple[list[int], list[float]]


@lru_cache
def _get_model() -> BGEM3FlagModel:
    settings = get_settings().text_embedding
    return BGEM3FlagModel(
        settings.model_name,
        devices=[settings.device],
        use_fp16=settings.device != "cpu",
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


# Needed to cache embeddings to avoid repeated computation for the same text because QdrantVectorStore takes embed_model and sparse_doc_fn arguments separately, which causes computation of embeddings twice for the same text when using the same embed_model and sparse_doc_fn.
_cache: dict[str, dict[str, Any]] = {}


def _encode(texts: list[str]) -> list[dict[str, Any]]:
    missing = [t for t in texts if t not in _cache]

    if missing:
        output = _get_model().encode(sentences=missing)

        for text, dense_vec, lexical_weights in zip(
            missing, output["dense_vecs"], output["lexical_weights"]
        ):
            _cache[text] = {"dense": dense_vec.tolist(), "sparse": lexical_weights}

    return [_cache[t] for t in texts]


def _to_sparse_vector(lexical_weights: defaultdict[str, np.floating]) -> SparseVector:
    indices = list(map(int, lexical_weights.keys()))
    values = list(map(float, lexical_weights.values()))
    return indices, values


class BGEM3Embedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "BGEM3Embedding"

    def _get_text_embedding(self, text: str) -> list[float]:
        return _encode([text])[0]["dense"]

    def _get_text_embedding(self, texts: list[str]) -> list[list[float]]:
        return [e["dense"] for e in _encode(texts)]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def bge_m3_sparse_doc_fn(texts: list[str]) -> BatchSparseEncoding:
    pairs = [_to_sparse_vector(e["sparse"]) for e in _encode(texts)]

    if not pairs:
        return [], []

    indices, values = zip(*pairs)
    return list(indices), list(values)


# symmetric - same model for query and doc embeddings
bge_m3_sparse_query_fn = bge_m3_sparse_doc_fn
