"""Weaviate (embedded) schema, loading, and search.

Two collections:
- Product: one object per product (title + features + description).
- ReviewChunk: one object per review chunk, with product metadata copied on so
  filters like "under $50" or "Switch accessories" work without a join.

Object UUIDs derive from our own IDs, so reloading overwrites instead of duplicating.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5

from review_rag_assistant.config import settings

PRODUCT = "Product"
REVIEW = "ReviewChunk"

Mode = Literal["vector", "bm25", "hybrid"]

_TEXT = wvc.config.DataType.TEXT
_NUM = wvc.config.DataType.NUMBER
_INT = wvc.config.DataType.INT
_BOOL = wvc.config.DataType.BOOL
_DATE = wvc.config.DataType.DATE


def _prop(name: str, dtype, searchable: bool = False, keyword: bool = False) -> wvc.config.Property:
    # Only `search_text` is BM25-indexed. Keyword-tokenized fields match exactly, for filters.
    return wvc.config.Property(
        name=name,
        data_type=dtype,
        index_searchable=searchable,
        tokenization=wvc.config.Tokenization.FIELD if keyword else None,
    )


PRODUCT_PROPS = [
    _prop("parent_asin", _TEXT, keyword=True),
    _prop("title", _TEXT),
    _prop("search_text", _TEXT, searchable=True),
    _prop("price", _NUM),
    _prop("average_rating", _NUM),
    _prop("rating_number", _INT),
    _prop("store", _TEXT, keyword=True),
    _prop("platform", _TEXT, keyword=True),
    _prop("product_type", _TEXT, keyword=True),
]

REVIEW_PROPS = [
    _prop("chunk_id", _TEXT, keyword=True),
    _prop("review_id", _TEXT, keyword=True),
    _prop("chunk_index", _INT),
    _prop("text", _TEXT),
    _prop("search_text", _TEXT, searchable=True),
    _prop("review_title", _TEXT),
    _prop("rating", _NUM),
    _prop("review_date", _DATE),
    _prop("helpful_vote", _INT),
    _prop("verified_purchase", _BOOL),
    # Copied from the product
    _prop("parent_asin", _TEXT, keyword=True),
    _prop("product_title", _TEXT),
    _prop("price", _NUM),
    _prop("average_rating", _NUM),
    _prop("platform", _TEXT, keyword=True),
    _prop("product_type", _TEXT, keyword=True),
]


@contextmanager
def connect() -> Iterator[weaviate.WeaviateClient]:
    client = weaviate.connect_to_embedded(
        version=settings.weaviate_version,
        persistence_data_path=str(settings.weaviate_data_dir),
        environment_variables={"LOG_LEVEL": "warning"},
    )
    try:
        yield client
    finally:
        client.close()


def create_schema(client: weaviate.WeaviateClient, recreate: bool = False) -> None:
    for name, props in ((PRODUCT, PRODUCT_PROPS), (REVIEW, REVIEW_PROPS)):
        if client.collections.exists(name):
            if not recreate:
                continue
            client.collections.delete(name)
        client.collections.create(
            name,
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            properties=props,
        )


def upsert(client: weaviate.WeaviateClient, collection: str, key: str, rows: list[dict], vectors) -> int:
    """Insert or overwrite objects. Returns the number of failures."""
    col = client.collections.use(collection)
    with col.batch.fixed_size(batch_size=500) as batch:
        for row, vec in zip(rows, vectors, strict=True):
            batch.add_object(properties=row, vector=list(map(float, vec)), uuid=generate_uuid5(row[key]))
    failed = col.batch.failed_objects
    return len(failed)


@dataclass
class Filters:
    """Structured constraints. None means no constraint."""

    max_price: float | None = None
    min_price: float | None = None
    min_average_rating: float | None = None
    platform: str | None = None
    product_type: str | None = None
    parent_asin: str | None = None
    # Review-level only
    max_review_rating: float | None = None
    min_review_rating: float | None = None

    def to_weaviate(self, collection: str):
        f = wvc.query.Filter
        parts = []
        if self.max_price is not None:
            parts.append(f.by_property("price").less_or_equal(self.max_price))
        if self.min_price is not None:
            parts.append(f.by_property("price").greater_or_equal(self.min_price))
        if self.min_average_rating is not None:
            parts.append(f.by_property("average_rating").greater_or_equal(self.min_average_rating))
        if self.platform:
            parts.append(f.by_property("platform").equal(self.platform))
        if self.product_type:
            parts.append(f.by_property("product_type").equal(self.product_type))
        if self.parent_asin:
            parts.append(f.by_property("parent_asin").equal(self.parent_asin))
        if collection == REVIEW:
            if self.max_review_rating is not None:
                parts.append(f.by_property("rating").less_or_equal(self.max_review_rating))
            if self.min_review_rating is not None:
                parts.append(f.by_property("rating").greater_or_equal(self.min_review_rating))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else f.all_of(parts)


@dataclass
class Hit:
    collection: str
    id: str  # parent_asin for products, chunk_id for review chunks
    properties: dict
    score: float  # first-stage score (hybrid, bm25, or vector similarity)
    rerank_score: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        return self.properties["search_text"]


def search(
    client: weaviate.WeaviateClient,
    collection: str,
    query: str,
    query_vector: list[float] | None,
    mode: Mode = "hybrid",
    limit: int = 30,
    filters: Filters | None = None,
    alpha: float = 0.5,
) -> list[Hit]:
    col = client.collections.use(collection)
    wf = filters.to_weaviate(collection) if filters else None
    meta = wvc.query.MetadataQuery(score=True, distance=True)
    if mode == "hybrid":
        res = col.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            query_properties=["search_text"],
            fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
            limit=limit,
            filters=wf,
            return_metadata=meta,
        )
    elif mode == "bm25":
        res = col.query.bm25(
            query=query, query_properties=["search_text"], limit=limit, filters=wf, return_metadata=meta
        )
    else:
        res = col.query.near_vector(near_vector=query_vector, limit=limit, filters=wf, return_metadata=meta)

    key = "chunk_id" if collection == REVIEW else "parent_asin"
    hits = []
    for o in res.objects:
        score = o.metadata.score if o.metadata.score is not None else 1 - (o.metadata.distance or 0)
        hits.append(Hit(collection=collection, id=o.properties[key], properties=o.properties, score=float(score)))
    return hits


def fetch_products(client: weaviate.WeaviateClient, asins: list[str]) -> list[Hit]:
    """Look up product listings by parent_asin, preserving the given order."""
    if not asins:
        return []
    res = client.collections.use(PRODUCT).query.fetch_objects(
        filters=wvc.query.Filter.by_property("parent_asin").contains_any(asins), limit=len(asins)
    )
    by_asin = {o.properties["parent_asin"]: o.properties for o in res.objects}
    return [Hit(collection=PRODUCT, id=a, properties=by_asin[a], score=0.0) for a in asins if a in by_asin]


def distinct_values(client: weaviate.WeaviateClient, prop: str) -> list[str]:
    """Distinct values of a keyword property on Product (used to tell the router valid platforms/types)."""
    res = client.collections.use(PRODUCT).aggregate.over_all(
        group_by=wvc.aggregate.GroupByAggregate(prop=prop, limit=200)
    )
    return sorted(g.grouped_by.value for g in res.groups if g.grouped_by.value)
