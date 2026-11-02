"""Local embedding and reranking models, loaded once per process."""

from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from review_rag_assistant.config import settings

# bge v1.5 models retrieve better when short queries carry this instruction. Passages get no prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    model = SentenceTransformer(settings.embed_model, device=settings.device)
    if settings.device == "cuda":
        model.half()
    return model


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    kwargs = {"torch_dtype": torch.float16} if settings.device == "cuda" else {}
    return CrossEncoder(settings.rerank_model, device=settings.device, max_length=512, model_kwargs=kwargs)


def embed_passages(texts: list[str], batch_size: int = 128) -> np.ndarray:
    return embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 1000
    )


def embed_query(query: str) -> list[float]:
    return embedder().encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()


def count_tokens(text: str) -> int:
    return len(embedder().tokenizer.encode(text, add_special_tokens=False))


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    return reranker().predict([(query, p) for p in passages], batch_size=32, show_progress_bar=False).tolist()
