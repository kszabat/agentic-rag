from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle

from agentic_rag.embeddings.colpali_embedding import colpali_embed_query
from agentic_rag.vector_store.qdrant_manager import (
    IMAGE_VECTOR_NAME,
    RagType,
    collection_name,
    get_qdrant_client,
)


class ColPaliRetriever(BaseRetriever):
    def __init__(self, kb_name: str, similarity_top_k: int = 5) -> None:
        self._collection = collection_name(kb_name=kb_name, rag_type=RagType.IMAGE)
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_vectors = colpali_embed_query(query=query_bundle.query_str)

        results = get_qdrant_client().query_points(
            collection_name=self._collection,
            query=query_vectors,
            using=IMAGE_VECTOR_NAME,
            limit=self._similarity_top_k,
            with_payload=True,
        )

        nodes_with_scores: list[NodeWithScore] = []
        for point in results.points:
            node = ImageNode(
                image_path=point.payload.get("image_path"),
                metadata=point.payload,
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=point.score))

        return nodes_with_scores


def get_colpali_retriever(kb_name: str, similarity_top_k: int = 5) -> BaseRetriever:
    return ColPaliRetriever(kb_name, similarity_top_k=similarity_top_k)
