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

from agentic_rag.llm.factory import get_llm
from agentic_rag.retrieval.reranker import get_reranker
from agentic_rag.retrieval.text_retriever import get_text_retriever
from agentic_rag.vector_store.qdrant_manager import RagType

TEXT_QA_PROMPT_STR = """Answer the question based only on the context below, and if the question can't be answered based on the context just say that you don't know - avoid fabricating an answer.

Context:
{{context_str}}

User Question: {{query_str}}

Response:"""

IMAGE_QA_INSTRUCTION = """User Question: {{query_str}}

Below are the document pages that may contain the answer to the question. If they are relevant, use them to answer the question. If they are not relevant, just say that you don't know - do not make things up.
"""

TEXT_QA_TEMPLATE = RichPromptTemplate(template_str=TEXT_QA_PROMPT_STR)
IMAGE_QA_TEMPLATE = RichPromptTemplate(template_str=IMAGE_QA_INSTRUCTION)

DEFAULT_TOP_K = {RagType.TEXT: 10, RagType.IMAGE: 5}


class RetrieverEvent(Event):
    nodes: list[NodeWithScore]


class RerankEvent(Event):
    nodes: list[NodeWithScore]


class RagWorkflow(Workflow):
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        query = ev.get("query")
        kb_name = ev.get("kb_name")
        mode = ev.get("mode", RagType.TEXT)
        if not query or not kb_name:
            return None

        await ctx.store.set("query", query)
        await ctx.store.set("mode", mode)

        similarity_top_k = ev.get("similarity_top_k", DEFAULT_TOP_K[mode])

        retriever = (
            get_text_retriever(kb_name, similarity_top_k=similarity_top_k)
            if mode is RagType.TEXT
            else get_text_retriever(
                kb_name, similarity_top_k=similarity_top_k, mode=RagType.IMAGE
            )
        )
        nodes = retriever.retrieve(query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        query = await ctx.store.get("query")
        reranker = get_reranker()
        nodes = reranker.postprocess_nodes(
            ev.nodes, query_bundle=QueryBundle(query_str=query)
        )
        return RerankEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        query = await ctx.store.get("query")
        mode = await ctx.store.get("mode")

        message = (
            self._build_text_message(query, ev.nodes)
            if mode is RagType.TEXT
            else self._build_image_message(query, ev.nodes)
        )

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
    workflow = RagWorkflow(timeout=120)
    result = await workflow.run(query=query, kb_name=kb_name, mode=RagType.TEXT)
    return str(result)


async def run_image_rag(kb_name: str, query: str) -> str:
    workflow = RagWorkflow(timeout=120)
    result = await workflow.run(query=query, kb_name=kb_name, mode=RagType.IMAGE)
    return str(result)
