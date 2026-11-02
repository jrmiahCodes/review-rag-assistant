"""Command line entry point: `uv run rra --help`."""

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

from review_rag_assistant.config import settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "weaviate"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def select() -> None:
    """Filter raw Amazon data down to the products and reviews we index."""
    from review_rag_assistant.ingest.select import select as run

    run(settings)


@app.command()
def index(recreate: bool = typer.Option(False, help="Drop and rebuild the Weaviate collections.")) -> None:
    """Chunk, embed, and load products and reviews into Weaviate."""
    from review_rag_assistant.ingest.index import index as run

    run(settings, recreate=recreate)


@app.command()
def ask(
    question: str,
    generator: str = typer.Option(
        None, "--generator", "-g", help="e.g. ollama:qwen3.5:4b or anthropic:claude-sonnet-5-5"
    ),
    mode: str = typer.Option("hybrid", help="First-stage search: hybrid, vector, or bm25"),
    rerank: bool = typer.Option(True, help="Rerank candidates with the cross-encoder"),
    show_sources: bool = typer.Option(True, help="Print the retrieved sources"),
) -> None:
    """Answer a question with citations."""
    from rich.table import Table

    from review_rag_assistant import store
    from review_rag_assistant.answer import answer_events, router_tools
    from review_rag_assistant.llm import get_llm
    from review_rag_assistant.retrieve import RetrievalConfig

    llm = get_llm(generator)
    with store.connect() as client:
        tools = router_tools(store.distinct_values(client, "platform"), store.distinct_values(client, "product_type"))
        for ev in answer_events(question, llm, client, tools, RetrievalConfig(mode=mode, rerank=rerank)):
            if ev["type"] == "plan":
                for c in ev["calls"]:
                    console.print(f"[dim]route -> {c['name']}({c['arguments']})[/dim]")
            elif ev["type"] == "sources" and show_sources:
                t = Table(show_header=True, header_style="dim", box=None)
                for col in ("#", "kind", "rerank", "product", "snippet"):
                    t.add_column(col)
                for s in ev["sources"]:
                    rr = f"{s['rerank_score']:.2f}" if s["rerank_score"] is not None else "-"
                    t.add_row(str(s["n"]), s["kind"], rr, s["title"][:40], s["text"][:60].replace("\n", " "))
                console.print(t)
                console.print()
            elif ev["type"] == "token":
                console.print(ev["text"], end="", markup=False, highlight=False)
            elif ev["type"] == "done":
                tm = ev["timings_ms"]
                console.print(
                    f"\n\n[dim]{ev['model']} | route {tm['route']:.0f}ms | retrieve {tm['retrieve']:.0f}ms | "
                    f"first token {tm['first_token']:.0f}ms | total {tm['total']:.0f}ms | "
                    f"tokens in/out {ev['usage']['input_tokens']}/{ev['usage']['output_tokens']}[/dim]"
                )


@app.command("eval-retrieval")
def eval_retrieval() -> None:
    """Score BM25, vector, hybrid, and hybrid+rerank on the golden set (no LLM calls)."""
    from rich.table import Table

    from review_rag_assistant import evals, store

    items = evals.load_golden()
    with store.connect() as client:
        result = evals.retrieval_eval(client, items)
    path = evals.save("retrieval", "all", result)
    t = Table(title=f"Retrieval ({len(items)} golden items)")
    cols = ["strict_hit@5", "strict_hit@10", "strict_mrr", "lenient_hit@5", "lenient_mrr", "p50_latency_ms"]
    t.add_column("config")
    for c in cols:
        t.add_column(c, justify="right")
    for name, s in result["summary"].items():
        t.add_row(name, *(f"{s[c]:.0f}" if c.endswith("ms") else f"{s[c]:.3f}" for c in cols))
    console.print(t)
    console.print(f"[dim]saved {path.relative_to(settings.processed_dir.parents[1])}[/dim]")


@app.command("eval-router")
def eval_router(generator: str = typer.Option(None, "--generator", "-g")) -> None:
    """Check the router's tool choice and filter extraction on the golden set."""
    from review_rag_assistant import evals, store

    items = evals.load_golden()
    with store.connect() as client:
        result = evals.router_eval(client, items, generator)
    path = evals.save("router", result["summary"]["generator"], result)
    for k, v in result["summary"].items():
        console.print(f"{k:24} {v:.3f}" if isinstance(v, float) else f"{k:24} {v}")
    for r in result["rows"]:
        if not r["tool_ok"] or r["args_ok"] is False or r["extra_filters"]:
            calls = "; ".join(f"{c['name']}({c['arguments']})" for c in r["calls"])
            console.print(f"[dim]{r['id']}: {calls}[/dim]", markup=True, highlight=False)
    console.print(f"[dim]saved {path.relative_to(settings.processed_dir.parents[1])}[/dim]")


@app.command("eval-generation")
def eval_generation(
    generator: str = typer.Option(None, "--generator", "-g"),
    judge: bool = typer.Option(True, help="Score answers with the Claude judge (needs ANTHROPIC_API_KEY)."),
    limit: int = typer.Option(0, help="Only run the first N items (0 = all)."),
) -> None:
    """Run the full pipeline on the golden set and score the answers."""
    from review_rag_assistant import evals, store

    items = evals.load_golden()
    if limit:
        items = items[:limit]
    with store.connect() as client:
        result = evals.generation_eval(client, items, generator, use_judge=judge)
    path = evals.save("generation", result["summary"]["generator"], result)
    for k, v in result["summary"].items():
        console.print(f"{k:32} {v:.3f}" if isinstance(v, float) else f"{k:32} {v}")
    console.print(f"[dim]saved {path.relative_to(settings.processed_dir.parents[1])}[/dim]")


@app.command()
def serve(port: int = 8000) -> None:
    """Run the API (and the built web UI, if present) on localhost."""
    import uvicorn

    uvicorn.run("review_rag_assistant.server:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
