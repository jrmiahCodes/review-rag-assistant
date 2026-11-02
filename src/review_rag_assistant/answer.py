"""Question -> route -> retrieve -> cited answer.

The pipeline yields events so the CLI and the web UI can show each stage as it
happens: the router's plan, the sources found, answer tokens, then timings.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import weaviate

from review_rag_assistant import store
from review_rag_assistant.llm import LLM, ToolCall, ToolSpec
from review_rag_assistant.retrieve import RetrievalConfig, retrieve, sort_products
from review_rag_assistant.store import Filters, Hit

log = logging.getLogger(__name__)

PRODUCTS_K = 6
REVIEWS_K = 8

ROUTER_SYSTEM = """You plan searches for a shopping assistant over Amazon video game products and their customer reviews (data up to 2023).

Call one or both tools:
- search_products: when the user wants product suggestions, comparisons, prices, or ratings, or gives constraints like price, platform, or "best rated".
- search_reviews: when the user asks what buyers think, about problems, durability, comfort, quality, or experiences with a product or kind of product.
For questions like "best X under $Y and do people complain about it", call both.

Rules:
- Only set a filter the user actually stated or clearly implied. Leave the rest out.
- Put the product name, if any, in the search_reviews query.
- Write queries as short descriptive phrases, not questions."""

ANSWER_SYSTEM = """You answer questions about video game products using ONLY the numbered sources provided. The sources are Amazon product listings and customer reviews from up to 2023.

Rules:
1. Cite every factual claim with source numbers in square brackets, like [2] or [1][4]. Never cite a source that does not support the claim.
2. If the sources do not answer the question, say so plainly and briefly. Do not fill gaps with outside knowledge.
3. The reviews you see were retrieved because they match the question, so they over-represent that topic. When a PRODUCT listing is present, state its overall rating and rating count to put the reviews in context (e.g. "several reviewers report drift [3][4], though the controller averages 4.7/5 across 40,352 ratings [1]"). Say "some reviewers" or "several reviewers", not "most" or "many", unless the sources show it.
4. Prices and ratings come from the product listings. Mention them when relevant.
5. Be concise: a short direct answer, then supporting detail. Use a bulleted list for multiple products."""


def router_tools(platforms: list[str], product_types: list[str]) -> list[ToolSpec]:
    common = {
        "max_price": {"type": "number", "description": "Maximum price in USD"},
        "platform": {"type": "string", "enum": platforms, "description": "Gaming platform"},
        "product_type": {"type": "string", "enum": product_types, "description": "Kind of product"},
    }
    return [
        ToolSpec(
            name="search_products",
            description="Find products by description with optional structured filters and sorting.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What kind of product, e.g. 'wireless headset with mic'",
                    },
                    **common,
                    "min_price": {"type": "number", "description": "Minimum price in USD"},
                    "min_average_rating": {"type": "number", "description": "Minimum average star rating, 1-5"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "rating", "popularity", "price_low", "price_high"],
                        "description": "Order of results. Use rating for 'best rated', popularity for 'most popular'.",
                    },
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="search_reviews",
            description="Search customer reviews for opinions, problems, and experiences.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to find in reviews, including the product name if any",
                    },
                    **common,
                    "max_review_rating": {
                        "type": "number",
                        "description": "Only reviews with at most this many stars, e.g. 2 for complaints",
                    },
                    "min_review_rating": {
                        "type": "number",
                        "description": "Only reviews with at least this many stars, e.g. 4 for praise",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@dataclass
class Source:
    n: int
    kind: str  # "product" | "review"
    id: str
    parent_asin: str
    title: str  # product title
    text: str
    meta: dict
    score: float
    rerank_score: float | None


def _clean_filters(args: dict, allowed: set[str]) -> Filters:
    vals = {k: v for k, v in args.items() if k in allowed and v not in (None, "", 0)}
    for k in ("max_price", "min_price", "min_average_rating", "max_review_rating", "min_review_rating"):
        if k in vals:
            try:
                vals[k] = float(vals[k])
            except (TypeError, ValueError):
                vals.pop(k)
    return Filters(**vals)


PRODUCT_FILTERS = {"max_price", "min_price", "min_average_rating", "platform", "product_type"}
REVIEW_FILTERS = {"max_price", "platform", "product_type", "max_review_rating", "min_review_rating"}


def plan(llm: LLM, question: str, tools: list[ToolSpec]) -> list[ToolCall]:
    try:
        calls = [c for c in llm.tool_calls(ROUTER_SYSTEM, question, tools) if c.name in {t.name for t in tools}]
    except Exception:  # a router failure degrades to plain review search instead of failing the request
        log.warning("router failed; falling back to review search", exc_info=True)
        calls = []
    return calls or [ToolCall("search_reviews", {"query": question})]


def execute(client: weaviate.WeaviateClient, call: ToolCall, cfg: RetrievalConfig) -> tuple[list[Hit], dict]:
    query = str(call.arguments.get("query") or "").strip()
    if call.name == "search_products":
        filters = _clean_filters(call.arguments, PRODUCT_FILTERS)
        res = retrieve(client, store.PRODUCT, query, k=PRODUCTS_K * 2, filters=filters, config=cfg)
        hits = sort_products(res.hits, call.arguments.get("sort_by") or "relevance", PRODUCTS_K)
    else:
        filters = _clean_filters(call.arguments, REVIEW_FILTERS)
        res = retrieve(client, store.REVIEW, query, k=REVIEWS_K, filters=filters, config=cfg)
        hits = res.hits
    return hits, res.timings_ms


LISTINGS_FOR_REVIEWS = 2


def add_listings_for_reviews(client: weaviate.WeaviateClient, hits: list[Hit]) -> list[Hit]:
    """Put the product listing for the most-reviewed products first.

    A search for "stick drift" returns only drift complaints. Without the listing's
    overall rating (e.g. 4.7 from 40K ratings) the generator can't tell a common
    complaint from a vocal minority.
    """
    have = {h.id for h in hits if h.collection == store.PRODUCT}
    counts: dict[str, int] = {}
    for h in hits:
        if h.collection == store.REVIEW and (a := h.properties["parent_asin"]) not in have:
            counts[a] = counts.get(a, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)[:LISTINGS_FOR_REVIEWS]
    return store.fetch_products(client, top) + hits


def to_sources(hits: list[Hit]) -> list[Source]:
    sources, seen = [], set()
    for h in hits:
        if h.id in seen:
            continue
        seen.add(h.id)
        p = h.properties
        if h.collection == store.PRODUCT:
            meta = {
                k: p.get(k) for k in ("price", "average_rating", "rating_number", "platform", "product_type", "store")
            }
            src = Source(
                len(sources) + 1, "product", h.id, h.id, p["title"], p["search_text"], meta, h.score, h.rerank_score
            )
        else:
            meta = {
                k: p.get(k)
                for k in ("rating", "review_title", "helpful_vote", "verified_purchase", "price", "average_rating")
            }
            meta["review_date"] = str(p.get("review_date"))[:10]
            src = Source(
                len(sources) + 1,
                "review",
                h.id,
                p["parent_asin"],
                p["product_title"],
                p["text"],
                meta,
                h.score,
                h.rerank_score,
            )
        sources.append(src)
    return sources


def format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        m = s.meta
        if s.kind == "product":
            head = (
                f"[{s.n}] PRODUCT: {s.title}\n"
                f"Price: ${m['price']:.2f} | Rating: {m['average_rating']}/5 from {m['rating_number']:,} ratings | "
                f"Platform: {m['platform']} | Type: {m['product_type']}"
            )
            body = s.text.split("\n", 2)[-1][:800]  # drop title/category lines already in the head
        else:
            head = f'[{s.n}] REVIEW of {s.title[:120]}\n{m["rating"]:.0f}/5 stars | "{m["review_title"]}" | {m["review_date"]}'
            body = s.text
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def answer_events(
    question: str,
    llm: LLM,
    client: weaviate.WeaviateClient,
    tools: list[ToolSpec],
    cfg: RetrievalConfig | None = None,
    router_llm: LLM | None = None,
) -> Iterator[dict]:
    cfg = cfg or RetrievalConfig()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    calls = plan(router_llm or llm, question, tools)
    timings["route"] = (time.perf_counter() - t0) * 1000
    yield {"type": "plan", "calls": [asdict(c) for c in calls]}

    t = time.perf_counter()
    hits: list[Hit] = []
    for call in calls:
        call_hits, _ = execute(client, call, cfg)
        hits.extend(call_hits)
    sources = to_sources(add_listings_for_reviews(client, hits))
    timings["retrieve"] = (time.perf_counter() - t) * 1000
    yield {"type": "sources", "sources": [asdict(s) for s in sources]}

    t = time.perf_counter()
    user = (
        f"Question: {question}\n\nSources:\n{format_context(sources)}"
        if sources
        else f"Question: {question}\n\nSources: (none found)"
    )
    first_token = None
    for token in llm.stream(ANSWER_SYSTEM, user):
        if first_token is None:
            first_token = (time.perf_counter() - t) * 1000
        yield {"type": "token", "text": token}
    timings["generate"] = (time.perf_counter() - t) * 1000
    timings["first_token"] = first_token or 0.0
    timings["total"] = (time.perf_counter() - t0) * 1000
    yield {"type": "done", "timings_ms": timings, "usage": asdict(llm.last_usage), "model": llm.name}


def answer(question: str, llm: LLM, client: weaviate.WeaviateClient, tools: list[ToolSpec], **kw) -> dict:
    """Non-streaming wrapper: collect the events into one result dict (used by evals)."""
    out = {"question": question, "answer": ""}
    for ev in answer_events(question, llm, client, tools, **kw):
        if ev["type"] == "token":
            out["answer"] += ev["text"]
        else:
            out.update({k: v for k, v in ev.items() if k != "type"})
    return out
