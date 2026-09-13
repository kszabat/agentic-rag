from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever

from agentic_rag.embeddings.bge_m3 import BGEM3Embedding
from agentic_rag.ingestion.text_pipeline import get_text_vector_store


def get_text_retriever(
    kb_name: str,
    similarity_top_k: int = 5,
    sparse_top_k: int = 12,
    alpha: float = 0.75,
    hybrid_top_k: int | None = None,
) -> BaseRetriever:
    index = VectorStoreIndex.from_vector_store(
        vector_store=get_text_vector_store(kb_name),
        embed_model=BGEM3Embedding,
    )

    return index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=similarity_top_k,
        sparse_top_k=sparse_top_k,
        alpha=alpha,
        hybrid_top_k=hybrid_top_k,
    )
