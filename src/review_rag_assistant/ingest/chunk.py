"""Split reviews into retrievable chunks.

Most reviews fit in one chunk. Long ones are split on sentence boundaries into
windows of about `max_tokens`, with a small overlap so a point that spans the
boundary still appears whole in one chunk.

Each chunk gets a header naming the product. A review saying "the battery dies
in two hours" never names the product, so without the header neither BM25 nor
the embedding can tie it to "Xbox controller battery life".
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_TITLE_CHARS = 120


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    review_id: str
    chunk_index: int
    text: str  # the review text in this chunk, shown to users and the LLM
    search_text: str  # header + text, used for BM25 and embeddings


def split_text(text: str, count_tokens: Callable[[str], int], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack sentences into windows of at most max_tokens (a single huge sentence becomes its own window)."""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]
    if not sentences:
        return []
    sizes = [count_tokens(s) for s in sentences]
    if sum(sizes) <= max_tokens:
        return [" ".join(sentences)]

    windows: list[str] = []
    start = 0
    while start < len(sentences):
        end, used = start, 0
        while end < len(sentences) and (used + sizes[end] <= max_tokens or end == start):
            used += sizes[end]
            end += 1
        windows.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        # Step back over trailing sentences until we have ~overlap_tokens of overlap,
        # but always move forward by at least one sentence.
        back, carried = end, 0
        while back - 1 > start and carried + sizes[back - 1] <= overlap_tokens:
            back -= 1
            carried += sizes[back]
        start = back
    return windows


def header(product_title: str, platform: str | None, product_type: str | None, rating: float, review_title: str) -> str:
    title = (
        product_title
        if len(product_title) <= MAX_TITLE_CHARS
        else product_title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0] + "..."
    )
    kind = " ".join(p for p in (platform, product_type) if p)
    lines = [f"Product: {title}"]
    if kind:
        lines.append(f"Category: {kind}")
    lines.append(f"Review ({rating:.0f}/5 stars): {review_title}".rstrip(": "))
    return "\n".join(lines)


def chunk_review(
    review_id: str,
    review_title: str,
    review_text: str,
    rating: float,
    product_title: str,
    platform: str | None,
    product_type: str | None,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    head = header(product_title, platform, product_type, rating, review_title)
    pieces = split_text(review_text, count_tokens, max_tokens, overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{review_id}-{i}",
            review_id=review_id,
            chunk_index=i,
            text=piece,
            search_text=f"{head}\n{piece}",
        )
        for i, piece in enumerate(pieces)
    ]
