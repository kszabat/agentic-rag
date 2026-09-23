from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from google.genai.errors import ServerError
from rich.console import Console

from agentic_rag.agent.workflow import run_text_rag
from agentic_rag.ingestion.image_pipeline import ingest_image_document
from agentic_rag.ingestion.text_pipeline import ingest_text_document
from agentic_rag.vector_store.qdrant_manager import (
    RagType,
    collection_exists,
    collection_name,
    delete_kb,
    list_kbs,
)

app = typer.Typer(
    name="rag",
    help="Agentic RAG CLI",
    no_args_is_help=True,
)
kb_app = typer.Typer(help="Knowledge base management commands", no_args_is_help=True)
app.add_typer(kb_app, name="kb")

console = Console()


def _iter_input_files(path: Path, mode: RagType) -> list[Path]:
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )

    if mode == RagType.IMAGE:
        valid = [f for f in files if f.suffix.lower() == ".pdf"]
        for f in files:
            if f not in valid:
                console.print(
                    f"[yellow]Skipping[/yellow] {f.name} - image ingestion only supports PDF files  "
                )
        files = valid

    return files


def _require_text_kb(kb: str) -> None:
    if not collection_exists(collection_name(kb, RagType.TEXT)):
        available = ", ".join(list_kbs()) or "(no available knowledge bases)"
        console.print(
            f"[red]Knowledge base '{kb}' (text) does not exist.[/red] Available: {available}"
        )
        raise typer.Exit(code=1)


@app.command()
def ingest(
    path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the document or directory to ingest",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
    kb: str = typer.Option(
        ..., "--kb", help="Name of the knowledge base to ingest into"
    ),
    mode: RagType = typer.Option(  # noqa: B008
        RagType.TEXT, "--mode", help="Ingestion mode: text or image"
    ),
) -> None:
    files = _iter_input_files(path, mode)
    if not files:
        console.print(
            f"[red]No valid files found in {path} for mode {mode.value}[/red]"
        )
        raise typer.Exit(code=1)

    ingest_fn = ingest_text_document if mode == RagType.TEXT else ingest_image_document
    unit = "chunks" if mode == RagType.TEXT else "pages"

    for file in files:
        try:
            count = ingest_fn(kb_name=kb, file_path=file)
            console.print(
                f"[green]Ingested[/green] {count} {unit} from {file.name} into knowledge base '{kb}'"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error ingesting {file.name}: {e}[/red]")
            # console.print_exception(show_locals=False)
            raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Argument(
        ..., help="Query string to search the knowledge base"
    ),
    kb: str = typer.Option(..., "--kb", help="Name of the knowledge base to query"),
) -> None:
    _require_text_kb(kb)

    with console.status(f"[cyan]Querying knowledge base '{kb}'...[/cyan]"):
        try:
            answer = asyncio.run(run_text_rag(kb, question))
        except ServerError:
            console.print("[red]LLM Server is currently unavailable. Try again later.[/red]")
            raise typer.Exit(code=1)
    console.print(f"[green]Answer:[/green] {answer}")


@app.command()
def chat(
    kb: str = typer.Option(..., "--kb", help="Name of the knowledge base to chat with"),
) -> None:
    _require_text_kb(kb)

    console.print(f"[cyan]Starting chat session with knowledge base '{kb}'...[/cyan]")
    while True:
        try:
            question = typer.prompt("You")
        except (KeyboardInterrupt, EOFError):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        with console.status(f"[cyan]Querying knowledge base '{kb}'...[/cyan]"):
            try:
                answer = asyncio.run(run_text_rag(kb, question))
            except ServerError:
                console.print("[red]LLM Server is currently unavailable. Try again later.[/red]")
                continue
        console.print(f"[green]Answer:[/green] {answer}")


@kb_app.command("list")
def kb_list() -> None:
    kbs = list_kbs()
    if not kbs:
        console.print("[yellow]No knowledge bases found.[/yellow]")
        return
    console.print("[green]Available knowledge bases:[/green]")
    for kb in kbs:
        console.print(f"- {kb}")


@kb_app.command("delete")
def kb_delete(
    kb: str = typer.Argument(..., help="Name of the knowledge base to delete"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Confirm deletion without prompting"
    ),
) -> None:
    if not yes and not typer.confirm(
        f"Are you sure you want to delete the knowledge base '{kb}'?"
    ):
        raise typer.Abort()
    delete_kb(kb)
    console.print(f"[green]Knowledge base '{kb}' deleted successfully.[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
