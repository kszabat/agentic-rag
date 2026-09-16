from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle

from agentic_rag.config import RerankerMode, get_settings

JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"


@lru_cache
def _get_local_model() -> Any:
    from transformers import AutoModel

    settings = get_settings().reranker
    model = AutoModel.from_pretrained(
        settings.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to(settings.device).eval()
    return model


def _score_local(
    text_pairs: list[list[str]],
    image_pairs: list[list[str]],
    text_max_length: int,
    image_max_length: int,
) -> tuple[list[float], list[float]]:
    model = _get_local_model()
    text_scores = (
        model.compute_score(text_pairs, max_length=text_max_length, doc_type="text")
        if text_pairs
        else []
    )
    image_scores = (
        model.compute_score(image_pairs, max_length=image_max_length, doc_type="image")
        if image_pairs
        else []
    )
    return text_scores, image_scores


def _image_to_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _score_api(
    query: str, texts: list[str], images: list[str]
) -> tuple[list[float], list[float]]:
    settings = get_settings().reranker
    documents = [{"text": t} for t in texts] + [
        {"image": _image_to_base64(p)} for p in images
    ]
    if not documents:
        return [], []

    response = requests.post(
        JINA_RERANK_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        },
        json={
            "model": settings.model_name,
            "query": query,
            "documents": documents,
            "return_documents": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    results = response.json()["results"]

    scores_by_index = {r["index"]: r["relevance_score"] for r in results}
    n_texts = len(texts)
    text_scores = [scores_by_index[i] for i in range(n_texts)]
    image_scores = [scores_by_index[i] for i in range(n_texts, n_texts + len(images))]
    return text_scores, image_scores


class JinaM0Reranker(BaseNodePostprocessor):
    top_n: int = 5
    text_max_length: int = 1024
    image_max_length: int = 2048

    @classmethod
    def class_name(cls) -> str:
        return "JinaM0Reranker"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes

        settings = get_settings().reranker
        query = query_bundle.query_str

        text_indices = [
            i for i, n in enumerate(nodes) if not isinstance(n.node, ImageNode)
        ]
        image_indices = [
            i for i, n in enumerate(nodes) if isinstance(n.node, ImageNode)
        ]

        text_inputs = [nodes[i].node.get_content() for i in text_indices]
        image_inputs = [nodes[i].node.image_path for i in image_indices]

        if settings.mode == RerankerMode.LOCAL:
            text_pairs = [[query, t] for t in text_inputs]
            image_pairs = [[query, im] for im in image_inputs]
            text_scores, image_scores = _score_local(
                text_pairs, image_pairs, self.text_max_length, self.image_max_length
            )
        else:
            text_scores, image_scores = _score_api(query, text_inputs, image_inputs)

        scores: dict[int, float] = {}
        scores.update(zip(text_indices, text_scores))
        scores.update(zip(image_indices, image_scores))

        for i, node_with_score in enumerate(nodes):
            node_with_score.score = scores[i]

        nodes.sort(key=lambda n: n.score, reverse=True)
        return nodes[: self.top_n]


def get_reranker(top_n: int | None = None) -> JinaM0Reranker:
    settings = get_settings().reranker
    return JinaM0Reranker(top_n=top_n or settings.top_n)
