"""Two-stage retrieval: first-stage search in Weaviate, then cross-encoder rerank.

The first stage casts a wide net (30 candidates) cheaply. The reranker reads each
(query, passage) pair together, which is slower but much better at judging
relevance, so it only sees the shortlist.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

import weaviate

from review_rag_assistant import models, store
from review_rag_assistant.store import Filters, Hit, Mode

SortBy = Literal["relevance", "rating", "popularity", "price_low", "price_high"]


@dataclass
class RetrievalConfig:
    mode: Mode = "hybrid"
    rerank: bool = True
    candidates: int = 30
    alpha: float = 0.5  # hybrid weight: 0 = pure BM25, 1 = pure vector


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


def retrieve(
    client: weaviate.WeaviateClient,
    collection: str,
    query: str,
    k: int,
    filters: Filters | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    cfg = config or RetrievalConfig()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    qvec = models.embed_query(query) if cfg.mode != "bm25" else None
    timings["embed"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    limit = cfg.candidates if cfg.rerank else k
    hits = store.search(client, collection, query, qvec, mode=cfg.mode, limit=limit, filters=filters, alpha=cfg.alpha)
    timings["search"] = (time.perf_counter() - t) * 1000

    if cfg.rerank and hits:
        t = time.perf_counter()
        scores = models.rerank_scores(query, [h.search_text for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = s
        hits.sort(key=lambda h: h.rerank_score, reverse=True)
        timings["rerank"] = (time.perf_counter() - t) * 1000

    return RetrievalResult(hits=hits[:k], timings_ms=timings)


def sort_products(hits: list[Hit], sort_by: SortBy, keep: int) -> list[Hit]:
    """Re-order relevant products by a structured field. Relevance has already narrowed the set."""
    if sort_by == "relevance":
        return hits[:keep]
    key = {
        "rating": lambda h: (h.properties["average_rating"], h.properties["rating_number"]),
        "popularity": lambda h: h.properties["rating_number"],
        "price_low": lambda h: -h.properties["price"],
        "price_high": lambda h: h.properties["price"],
    }[sort_by]
    return sorted(hits, key=key, reverse=True)[:keep]
