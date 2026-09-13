from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle

from agentic_rag.config import get_settings


@lru_cache
def _get_model() -> Any:
    from transformers import AutoModel

    settings = get_settings().reranker
    model = AutoModel.from_pretrained(
        settings.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to(settings.device).eval()
    return model


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
        if query_bundle is None or not nodes or query_bundle.query_str == "":
            return nodes

        model = _get_model()
        query = query_bundle.query_str

        text_indices = [
            i for i, n in enumerate(nodes) if not isinstance(n.node, ImageNode)
        ]
        image_indices = [
            i for i, n in enumerate(nodes) if isinstance(n.node, ImageNode)
        ]

        scores: dict[int, float] = {}

        if text_indices:
            pairs = [[query, nodes[i].node.get_content()] for i in text_indices]
            text_scores = model.compute_score(
                pairs, max_length=self.text_max_length, doc_type="text"
            )
            scores.update(zip(text_indices, text_scores))

        if image_indices:
            pairs = [[query, nodes[i].node.image_path] for i in image_indices]
            image_scores = model.compute_score(
                pairs, max_length=self.image_max_length, doc_type="image"
            )
            scores.update(zip(image_indices, image_scores))

        for i, node_with_score in enumerate(nodes):
            node_with_score.score = scores[i]

        nodes.sort(key=lambda n: n.score, reverse=True)
        return nodes[: self.top_n]


def get_reranker(top_n: int | None = None) -> JinaM0Reranker:
    settings = get_settings().reranker
    return JinaM0Reranker(top_n=top_n or settings.top_n)
