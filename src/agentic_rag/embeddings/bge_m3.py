from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from llama_index.core.embeddings import BaseEmbedding

from agentic_rag.config import get_settings

SparseVector = tuple[list[int], list[float]]


@lru_cache
def _get_model() -> BGEM3FlagModel:
    settings = get_settings().text_embedding
    kwargs: dict[str, Any] = {"use_fp16": settings.device != "cpu"}
    if settings.device != "cpu":
        kwargs["devices"] = [settings.device]
    return BGEM3FlagModel(
        settings.model_name,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        **kwargs,
    )

_cache: dict[str, dict[str, Any]] = {}


def _encode(texts: list[str]) -> list[dict[str, Any]]:
    missing = [t for t in texts if t not in _cache]
    if missing:
        output = _get_model().encode(missing)

        dense_vecs = np.atleast_2d(output["dense_vecs"])
        for text, dense_vec, lexical_weights in zip(
            missing, dense_vecs, output["lexical_weights"]
        ):
            _cache[text] = {"dense": dense_vec.tolist(), "sparse": lexical_weights}
    return [_cache[t] for t in texts]


def _to_sparse_vector(lexical_weights: dict[Any, Any]) -> SparseVector:
    indices = [int(token_id) for token_id in lexical_weights]
    values = [float(v) for v in lexical_weights.values()]
    return indices, values


class BGEM3Embedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "BGEM3Embedding"

    def _get_text_embedding(self, text: str) -> list[float]:
        return _encode([text])[0]["dense"]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [item["dense"] for item in _encode(texts)]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def bge_m3_sparse_doc_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    pairs = [_to_sparse_vector(item["sparse"]) for item in _encode(texts)]
    if not pairs:
        return [], []
    indices, values = zip(*pairs)
    return list(indices), list(values)


bge_m3_sparse_query_fn = bge_m3_sparse_doc_fn
