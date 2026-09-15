from __future__ import annotations

from llama_index.core.llms import ChatMessage
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

QA_PROMPT_STR = """Answer the question based only on the context below, and if the question can't be answered based on the context just say that you don't know - avoid fabricating an answer.

Context:
{{context_str}}

User Question: {{query_str}}

Response:"""

QA_TEMPLATE = RichPromptTemplate(template_str=QA_PROMPT_STR)


class RetrieverEvent(Event):
    nodes: list[NodeWithScore]


class RerankEvent(Event):
    nodes: list[NodeWithScore]


class TextRagWorkflow(Workflow):
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        query = ev.get("query")
        kb_name = ev.get("kb_name")
        if not query or not kb_name:
            return None

        await ctx.store.set("query", query)

        similarity_top_k = ev.get("similarity_top_k", 10)
        retriever = get_text_retriever(kb_name, similarity_top_k=similarity_top_k)
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
        context = "\n\n---\n\n".join(n.node.get_content() for n in ev.nodes)
        prompt = QA_TEMPLATE.format(context_str=context, query_str=query)

        response = await get_llm().achat([ChatMessage(role="user", content=prompt)])
        return StopEvent(result=response)


async def run_text_rag(kb_name: str, query: str) -> str:
    workflow = TextRagWorkflow(timeout=120)
    result = await workflow.run(query=query, kb_name=kb_name)
    return str(result)
