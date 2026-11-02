"""Chunk, embed, and load the selected data into Weaviate.

data/processed/{products,reviews}.parquet -> data/processed/chunks.parquet + Weaviate.
"""

import logging
import time
from datetime import UTC, datetime

import duckdb

from review_rag_assistant import models, store
from review_rag_assistant.config import Settings
from review_rag_assistant.ingest.chunk import chunk_review

log = logging.getLogger(__name__)

MAX_PRODUCT_TEXT_CHARS = 2000  # bge truncates at 512 tokens anyway


def product_search_text(p: dict) -> str:
    parts = [p["title"], f"Category: {p['platform']} {p['product_type'] or ''}".strip()]
    if p["features"]:
        parts.append(p["features"])
    if p["description"]:
        parts.append(p["description"])
    return "\n".join(parts)[:MAX_PRODUCT_TEXT_CHARS]


def _records(sql: str) -> list[dict]:
    """Run a query and return rows as dicts, with SQL NULLs as None (pandas would give NaN)."""
    df = duckdb.sql(sql).df()
    return df.astype(object).where(df.notna(), None).to_dict("records")


def build_chunks(settings: Settings) -> tuple[list[dict], list[dict]]:
    proc = settings.processed_dir
    products = _records(f"select * from '{proc / 'products.parquet'}'")
    reviews = _records(
        f"""
        select r.*, p.title as product_title, p.price, p.average_rating, p.platform, p.product_type
        from '{proc / "reviews.parquet"}' r join '{proc / "products.parquet"}' p using (parent_asin)
        """
    )

    product_rows = [
        {
            "parent_asin": p["parent_asin"],
            "title": p["title"],
            "search_text": product_search_text(p),
            "price": float(p["price"]),
            "average_rating": float(p["average_rating"]),
            "rating_number": int(p["rating_number"]),
            "store": p["store"],
            "platform": p["platform"],
            "product_type": p["product_type"],
        }
        for p in products
    ]

    chunk_rows = []
    for r in reviews:
        for c in chunk_review(
            review_id=r["review_id"],
            review_title=r["title"],
            review_text=r["text"],
            rating=r["rating"],
            product_title=r["product_title"],
            platform=r["platform"],
            product_type=r["product_type"],
            count_tokens=models.count_tokens,
            max_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ):
            chunk_rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "review_id": c.review_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "search_text": c.search_text,
                    "review_title": r["title"],
                    "rating": float(r["rating"]),
                    "review_date": datetime.combine(r["review_date"], datetime.min.time(), tzinfo=UTC),
                    "helpful_vote": int(r["helpful_vote"]),
                    "verified_purchase": bool(r["verified_purchase"]),
                    "parent_asin": r["parent_asin"],
                    "product_title": r["product_title"],
                    "price": float(r["price"]),
                    "average_rating": float(r["average_rating"]),
                    "platform": r["platform"],
                    "product_type": r["product_type"],
                }
            )
    return product_rows, chunk_rows


def index(settings: Settings, recreate: bool = False) -> dict[str, int]:
    t0 = time.time()
    product_rows, chunk_rows = build_chunks(settings)
    log.info("built %d products, %d review chunks in %.1fs", len(product_rows), len(chunk_rows), time.time() - t0)

    # Keep a flat copy of the chunks: the eval set is generated from it.
    import pandas as pd

    pd.DataFrame(chunk_rows).to_parquet(settings.processed_dir / "chunks.parquet", index=False)

    t1 = time.time()
    product_vecs = models.embed_passages([p["search_text"] for p in product_rows])
    chunk_vecs = models.embed_passages([c["search_text"] for c in chunk_rows])
    log.info("embedded %d texts in %.1fs", len(product_rows) + len(chunk_rows), time.time() - t1)

    t2 = time.time()
    with store.connect() as client:
        store.create_schema(client, recreate=recreate)
        failed_p = store.upsert(client, store.PRODUCT, "parent_asin", product_rows, product_vecs)
        failed_c = store.upsert(client, store.REVIEW, "chunk_id", chunk_rows, chunk_vecs)
        n_products = len(client.collections.use(store.PRODUCT))
        n_chunks = len(client.collections.use(store.REVIEW))
    log.info("loaded into Weaviate in %.1fs", time.time() - t2)

    counts = {
        "products_built": len(product_rows),
        "chunks_built": len(chunk_rows),
        "products_in_weaviate": n_products,
        "chunks_in_weaviate": n_chunks,
        "failed_objects": failed_p + failed_c,
    }
    for k, v in counts.items():
        log.info("%-24s %8d", k, v)
    if n_chunks != len(chunk_rows) or n_products != len(product_rows):
        log.warning("row count mismatch between built rows and Weaviate; rerun with --recreate")
    return counts
