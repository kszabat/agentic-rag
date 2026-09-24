# RAG CLI

A Retrieval-Augmented Generation (RAG) tool built on [Qdrant](https://qdrant.tech/) as the vector database, using [Docling's](https://docling.ai/)  node parser and document reader for document parsing, orchestrated with [LlamaIndex](https://www.llamaindex.ai/).

> **Note:** Only the text ingestion/query pipeline is currently implemented.

## Installation

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository and move into it:

```bash
   git clone https://github.com/kszabat/agentic-rag.git
   cd agentic-rag
```

3. Sync the project dependencies:

```bash
   uv sync
```

## Setup

1. Copy `.env.example` to `.env`:

```bash
   cp .env.example .env
```

2. Open `.env` and fill in / edit the required values.

# Commands
- `rag ingest <path> --kb <kb_name> [--mode text|image]` (obecnie tylko `text` jest wspierane)
- `rag query "<question>" --kb <kb_name>`
- `rag kb list`
- `rag kb delete <name> [--yes]`