from __future__ import annotations

from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import BaseModel, Field

from typing import Literal

from agentic_rag.llm.factory import get_llm
from agentic_rag.retrieval.reranker import get_reranker
from agentic_rag.retrieval.text_retriever import get_text_retriever
from agentic_rag.vector_store.qdrant_manager import RagType

MAX_ATTEMPTS = 2

DEFAULT_TOP_K = {RagType.TEXT: 10, RagType.IMAGE: 5}

TEXT_QA_PROMPT_STR = """Answer the question based only on the context below, and if the question can't be answered based on the context just say that you don't know - avoid fabricating an answer.

Context:
{{context_str}}

User Question: {{query_str}}

Response:"""

IMAGE_QA_INSTRUCTION = """User Question: {{query_str}}

Below are the document pages that may contain the answer to the question. If they are relevant, use them to answer the question. If they are not relevant, just say that you don't know - do not make things up.
"""

QUERY_PLAN_PROMPT_STR = """
You have to evaluate user query for a RAG system. Decide whether it is woth reformulating the query before retrieval (e.g. if it is too vague, colloquial, or lacks context) or if it is already precise enough to yield relevant results without modification. Do not reformulate unnecessarily.

User Query: {{query_str}}
"""


EVALUATE_PROMPT_STR = """Assess whether the provided context is sufficient to answer the question effectively. If it is NOT sufficient, choose a strategy for the next attempt:
- more_context: the context is on the right track but insufficient - use the same query but retrieve more results
- rewrite_query: the query was poorly phrased or too narrow - rephrase it
- hyde: there is a mismatch between the query vocabulary and the document vocabulary - write a short, hypothetical document excerpt or answer to use for a semantic search instead of the original query.

User Question: {{query_str}}

Context: 
{{context_str}}
"""


class QueryPlan(BaseModel):
    """
    Decision: Whether the query needs to be reformulated before retrieval, and if so, what the reformulated query should be.
    """

    needs_rewrite: bool = Field(
        description="Whether the query needs to be reformulated before retrieval."
    )
    rewritten_query: str | None = Field(
        default=None,
        description="The reformulated query - fill only if needs_rewrite is True. ",
    )


class RetrievalEvaluation(BaseModel):
    """
    Decision: Whether the retrieved context is sufficient to answer the question effectively, and if not, what strategy to use for the next attempt.
    """

    sufficient: bool = Field(
        description="Whether the retrieved context is sufficient to answer the question effectively."
    )
    strategy: Literal["more_context", "rewrite_query", "hyde"] = Field(
        default="more_context",
        description="The strategy to use for the next attempt - fill only if sufficient is False.",
    )
    new_query_or_passage: str | None = Field(
        default=None,
        description="For 'rewrite_query', the new query to use for the next attempt. For 'hyde', a hypothetical fragment of a document or answer to use for a semantic search instead of the original query. Empty for 'more_context' or if sufficient is True.",
    )


TEXT_QA_TEMPLATE = RichPromptTemplate(template_str=TEXT_QA_PROMPT_STR)
IMAGE_QA_TEMPLATE = RichPromptTemplate(template_str=IMAGE_QA_INSTRUCTION)
QUERY_PLAN_TEMPLATE = RichPromptTemplate(template_str=QUERY_PLAN_PROMPT_STR)
EVALUATE_TEMPLATE = RichPromptTemplate(template_str=EVALUATE_PROMPT_STR)


class SearchEvent(Event):
    search_text: str
    similarity_top_k: int
    attempt: int


class RetrieverEvent(Event):
    nodes: list[NodeWithScore]
    search_text: str
    similarity_top_k: int
    attempt: int


class RerankEvent(Event):
    nodes: list[NodeWithScore]
    search_text: str
    similarity_top_k: int
    attempt: int


class SynthesizeEvent(Event):
    nodes: list[NodeWithScore]


class RagWorkflow(Workflow):
    @step
    async def plan_query(self, ctx: Context, ev: StartEvent) -> SearchEvent | None:
        query = ev.get("query")
        kb_name = ev.get("kb_name")
        mode = ev.get("mode", RagType.TEXT)
        if not query or not kb_name:
            return None

        await ctx.store.set("query", query)
        await ctx.store.set("mode", mode)
        await ctx.store.set("kb_name", kb_name)

        plan = await get_llm().astructured_predict(
            QueryPlan, QUERY_PLAN_TEMPLATE, query_str=query
        )

        search_text = plan.rewritten_query if plan.needs_rewrite else query
        similarity_top_k = ev.get("similarity_top_k", DEFAULT_TOP_K[mode])

        return SearchEvent(
            search_text=search_text,
            similarity_top_k=similarity_top_k,
            attempt=0,
        )

    @step
    async def retrieve(self, ctx: Context, ev: SearchEvent) -> RetrieverEvent:
        mode = await ctx.store.get("mode")
        kb_name = await ctx.store.get("kb_name")

        if mode == RagType.TEXT:
            retriever = get_text_retriever(
                kb_name, similarity_top_k=ev.similarity_top_k
            )
        else:
            raise NotImplementedError("Image retrieval is not implemented yet.")

        nodes = retriever.retrieve(ev.search_text)
        return RetrieverEvent(
            nodes=nodes,
            search_text=ev.search_text,
            similarity_top_k=ev.similarity_top_k,
            attempt=ev.attempt,
        )

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        reranker = get_reranker()
        nodes = reranker.postprocess_nodes(
            ev.nodes, query_bundle=QueryBundle(query_str=ev.search_text)
        )
        return RerankEvent(
            nodes=nodes,
            search_text=ev.search_text,
            similarity_top_k=ev.similarity_top_k,
            attempt=ev.attempt,
        )

    @step
    async def evaluate(
        self, ctx: Context, ev: RerankEvent
    ) -> SynthesizeEvent | SearchEvent:
        mode = await ctx.store.get("mode")

        if mode == RagType.IMAGE:
            raise NotImplementedError("Image evaluation is not implemented yet.")
        if ev.attempt >= MAX_ATTEMPTS:
            return SynthesizeEvent(nodes=ev.nodes)

        original_query = await ctx.store.get("query")
        context_preview = "\n\n".join(n.node.get_content()[:800] for n in ev.nodes)

        evaluation = await get_llm().astructured_predict(
            RetrievalEvaluation,
            EVALUATE_TEMPLATE,
            query_str=original_query,
            context_str=context_preview,
        )

        if evaluation.sufficient:
            return SynthesizeEvent(nodes=ev.nodes)

        if evaluation.strategy == "more_context":
            next_search_text = ev.search_text
            next_top_k = ev.similarity_top_k * 2
        elif (
            evaluation.strategy in ("rewrite_query", "hyde")
            and evaluation.new_query_or_passage
        ):
            next_search_text = evaluation.new_query_or_passage
            next_top_k = ev.similarity_top_k
        else:
            next_search_text = ev.search_text
            next_top_k = ev.similarity_top_k

        return SearchEvent(
            search_text=next_search_text,
            similarity_top_k=next_top_k,
            attempt=ev.attempt + 1,
        )

    @step
    async def synthesize(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        query = await ctx.store.get("query")
        mode = await ctx.store.get("mode")

        if mode == RagType.TEXT:
            message = self._build_text_message(query, ev.nodes)
        else:
            raise NotImplementedError("Image synthesis is not implemented yet.")

        response = await get_llm().achat([message])
        return StopEvent(result=response)

    @staticmethod
    def _build_text_message(query: str, nodes: list[NodeWithScore]) -> ChatMessage:
        context = "\n\n---\n\n".join(n.node.get_content() for n in nodes)
        prompt = TEXT_QA_TEMPLATE.format(context_str=context, query_str=query)
        return ChatMessage(role="user", content=prompt)

    @staticmethod
    def _build_image_message(query: str, nodes: list[NodeWithScore]) -> ChatMessage:
        blocks: list[TextBlock | ImageBlock] = [
            TextBlock(text=IMAGE_QA_TEMPLATE.format(query_str=query))
        ]
        for node_with_score in nodes:
            node = node_with_score.node
            meta = node.metadata
            caption = f"--- Page {meta.get('page_number')} from {meta.get('doc_id')} document ---"
            blocks.append(TextBlock(text=caption))
            blocks.append(ImageBlock(path=node.image_path, image_mimetype="image/png"))

        return ChatMessage(role="user", content=blocks)


async def run_text_rag(kb_name: str, query: str) -> str:
    workflow = RagWorkflow(timeout=180)
    result = await workflow.run(query=query, kb_name=kb_name, mode=RagType.TEXT)
    return str(result)


async def run_image_rag(kb_name: str, query: str) -> str:
    workflow = RagWorkflow(timeout=180)
    result = await workflow.run(query=query, kb_name=kb_name, mode=RagType.IMAGE)
    return str(result)
