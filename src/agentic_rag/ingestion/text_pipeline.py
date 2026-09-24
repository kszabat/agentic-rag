from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.qdrant import QdrantVectorStore

_pipeline_options = PdfPipelineOptions()
_table_structure_options = TableStructureOptions()
_table_structure_options.mode = TableFormerMode.ACCURATE
_pipeline_options.generate_page_images = False
_pipeline_options.generate_picture_images = False
# _pipeline_options.images_scale =
_pipeline_options.do_table_structure = True
_pipeline_options.do_ocr = False
_pipeline_options.table_structure_options = _table_structure_options

_document_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

_reader = DoclingReader(
    doc_converter=_document_converter, export_type=DoclingReader.ExportType.JSON
)
_node_parser = DoclingNodeParser()

from agentic_rag.embeddings.bge_m3 import (
    BGEM3Embedding,
    bge_m3_sparse_doc_fn,
    bge_m3_sparse_query_fn,
)
from agentic_rag.vector_store.qdrant_manager import (
    TEXT_DENSE_VECTOR_NAME,
    TEXT_SPARSE_VECTOR_NAME,
    RagType,
    collection_name,
    get_qdrant_client,
)


def get_text_vector_store(kb_name: str) -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name=collection_name(kb_name=kb_name, rag_type=RagType.TEXT),
        client=get_qdrant_client(),
        enable_hybrid=True,
        dense_vector_name=TEXT_DENSE_VECTOR_NAME,
        sparse_vector_name=TEXT_SPARSE_VECTOR_NAME,
        sparse_doc_fn=bge_m3_sparse_doc_fn,
        sparse_query_fn=bge_m3_sparse_query_fn,
    )


def ingest_text_document(kb_name: str, file_path: str | Path) -> int:
    documents = _reader.load_data(file_path=str(file_path))
    nodes = _node_parser.get_nodes_from_documents(documents)

    storage_context = StorageContext.from_defaults(
        vector_store=get_text_vector_store(kb_name=kb_name)
    )

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=BGEM3Embedding(),
        show_progress=True,
    )

    return len(nodes)
