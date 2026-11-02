"""Select the products and reviews we index, using DuckDB over the raw JSONL.

Raw files (data/raw) -> data/processed/products.parquet, reviews.parquet.
Rerunning overwrites the outputs, so the step is idempotent.
"""

import logging
import time

import duckdb

from review_rag_assistant.config import Settings

log = logging.getLogger(__name__)

META_COLUMNS = {
    "parent_asin": "VARCHAR",
    "title": "VARCHAR",
    "average_rating": "DOUBLE",
    "rating_number": "BIGINT",
    "price": "VARCHAR",
    "store": "VARCHAR",
    "features": "VARCHAR[]",
    "description": "VARCHAR[]",
    "categories": "VARCHAR[]",
}

REVIEW_COLUMNS = {
    "parent_asin": "VARCHAR",
    "user_id": "VARCHAR",
    "rating": "DOUBLE",
    "title": "VARCHAR",
    "text": "VARCHAR",
    "timestamp": "BIGINT",
    "helpful_vote": "BIGINT",
    "verified_purchase": "BOOLEAN",
}


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"select count(*) from {table}").fetchall()[0][0]


def select(settings: Settings) -> dict[str, int]:
    """Run the selection and return row counts for each step."""
    meta_path = settings.raw_dir / f"meta_{settings.category}.jsonl"
    reviews_path = settings.raw_dir / f"{settings.category}.jsonl"
    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    counts: dict[str, int] = {}
    t0 = time.time()

    con.execute(
        "create table meta as select * from read_json(?, columns=$cols, format='newline_delimited')".replace(
            "$cols", str(META_COLUMNS)
        ),
        [str(meta_path)],
    )
    counts["meta_raw"] = _count(con, "meta")

    # Products: priced, enough ratings to be meaningful, top N by rating count.
    # Legacy consoles nest two levels deeper: [Video Games, Legacy Systems, Nintendo Systems, Wii, Games].
    # Gift cards and digital currency are excluded: their reviews are about code redemption, not products.
    con.execute(
        """
        create table products as
        with eligible as (
            select
                parent_asin,
                trim(title) as title,
                try_cast(price as double) as price,
                average_rating,
                rating_number,
                store,
                array_to_string(features, '\n') as features,
                array_to_string(description, '\n') as description,
                case when categories[2] = 'Legacy Systems' then categories[4] else categories[2] end as platform,
                case when categories[2] = 'Legacy Systems' then categories[5] else categories[3] end as product_type
            from meta
            where title is not null and length(trim(title)) > 0
              and not regexp_matches(title, '(?i)gift card|gift code|digital code|robux|v-bucks|points card|membership')
              and try_cast(price as double) > 0
              and rating_number >= ?
        )
        select * from eligible
        -- No category path usually means a misfiled item (books, DVDs) rather than a game product.
        where platform is not null
        qualify row_number() over (partition by parent_asin order by rating_number desc) = 1
        order by rating_number desc
        limit ?
        """,
        [settings.min_ratings, settings.max_products],
    )
    counts["products_selected"] = _count(con, "products")

    # Reviews for the selected products only. The semi join keeps the scan cheap.
    con.execute(
        "create table reviews_raw as select * from read_json(?, columns=$cols, format='newline_delimited') r "
        "where r.parent_asin in (select parent_asin from products)".replace("$cols", str(REVIEW_COLUMNS)),
        [str(reviews_path)],
    )
    counts["reviews_for_selected_products"] = _count(con, "reviews_raw")

    # Keep the most useful reviews per product: verified first, then most helpful, then longest.
    # review_id is a stable hash so reruns produce the same IDs (needed for idempotent upserts and evals).
    con.execute(
        """
        create table reviews as
        select
            md5(user_id || '|' || parent_asin || '|' || cast(timestamp as varchar)) as review_id,
            parent_asin,
            rating,
            coalesce(trim(title), '') as title,
            -- Review text carries HTML line breaks and a few entities; normalize them to plain text.
            trim(replace(replace(replace(replace(
                regexp_replace(text, '<br\\s*/?>', '\n', 'gi'),
                '&quot;', '"'), '&#34;', '"'), '&amp;', '&'), '&#39;', '''')) as text,
            to_timestamp(timestamp / 1000)::date as review_date,
            helpful_vote,
            verified_purchase
        from reviews_raw
        where text is not null and length(trim(text)) >= ?
        qualify row_number() over (
            partition by parent_asin
            order by verified_purchase desc, helpful_vote desc, length(text) desc
        ) <= ?
        """,
        [settings.min_review_chars, settings.reviews_per_product],
    )
    # The same person can post the same review twice; drop exact duplicates.
    con.execute(
        "create table reviews_dedup as select * from reviews qualify row_number() over (partition by review_id) = 1"
    )
    counts["reviews_selected"] = _count(con, "reviews_dedup")

    con.execute(f"copy products to '{out / 'products.parquet'}' (format parquet)")
    con.execute(f"copy reviews_dedup to '{out / 'reviews.parquet'}' (format parquet)")

    for step, n in counts.items():
        log.info("%-32s %10d", step, n)
    log.info("selection finished in %.1fs", time.time() - t0)
    return counts
