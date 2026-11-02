"""HTTP API for the web UI. Run with `uv run rra serve`.

POST /api/ask streams pipeline events as server-sent events, one JSON object per
`data:` line, in the same shape the CLI consumes (plan, sources, token, done).
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from review_rag_assistant import store
from review_rag_assistant.answer import answer_events, router_tools
from review_rag_assistant.config import ROOT, settings
from review_rag_assistant.evals import RESULTS_DIR
from review_rag_assistant.llm import get_llm
from review_rag_assistant.retrieve import RetrievalConfig

log = logging.getLogger(__name__)
WEB_DIST = ROOT / "web" / "dist"

state: dict = {}
# The embedded Weaviate client and the GPU models are shared, so serialize requests.
pipeline_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from review_rag_assistant import models

    ctx = store.connect()
    client = ctx.__enter__()
    platforms = store.distinct_values(client, "platform")
    product_types = store.distinct_values(client, "product_type")
    state.update(
        client=client,
        platforms=platforms,
        product_types=product_types,
        tools=router_tools(platforms, product_types),
        llms={},
        counts={
            "products": len(client.collections.use(store.PRODUCT)),
            "review_chunks": len(client.collections.use(store.REVIEW)),
        },
    )
    models.embedder()  # load models at startup so the first question isn't slow
    models.reranker()
    try:
        yield
    finally:
        ctx.__exit__(None, None, None)


app = FastAPI(title="review-rag-assistant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    generator: str | None = None
    mode: str = Field("hybrid", pattern="^(hybrid|vector|bm25)$")
    rerank: bool = True


def _llm(spec: str | None):
    spec = spec or settings.generator
    if spec not in state["llms"]:
        state["llms"][spec] = get_llm(spec)
    return state["llms"][spec]


@app.get("/api/meta")
def meta() -> dict:
    available = [settings.generator]
    if settings.anthropic_api_key:
        available.append("anthropic:claude-sonnet-5-5")
    return {
        "counts": state["counts"],
        "platforms": state["platforms"],
        "product_types": state["product_types"],
        "default_generator": settings.generator,
        "generators": available,
        "data_through": "2023-09",
    }


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    try:
        llm = _llm(req.generator)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    cfg = RetrievalConfig(mode=req.mode, rerank=req.rerank)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def work() -> None:
        try:
            with pipeline_lock:
                for ev in answer_events(req.question, llm, state["client"], state["tools"], cfg):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:  # surface pipeline failures to the UI instead of a dropped stream
            log.exception("pipeline failed for %r", req.question)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while (ev := await queue.get()) is not None:
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/evals")
def evals() -> dict:
    """Latest saved summary per eval kind and generator."""
    latest: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []:
        _, kind, label = path.stem.split("_", 2)
        data = json.loads(path.read_text())
        latest[f"{kind}:{label}"] = {"kind": kind, "label": label, "file": path.name, "summary": data["summary"]}
    return {"results": list(latest.values())}


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        file = WEB_DIST / path
        return FileResponse(file if path and file.is_file() else Path(WEB_DIST / "index.html"))
