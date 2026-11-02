"""Local embedding and reranking models, loaded once per process."""

import os
from functools import lru_cache
from pathlib import Path

from review_rag_assistant.config import settings


def _use_offline_hub_if_cached() -> None:
    """Skip Hugging Face network checks when both models are already downloaded.

    Without this, every start makes HTTP calls to the Hub, which slows startup and fails
    a live demo on a bad connection. A fresh clone still downloads normally.
    """
    if "HF_HUB_OFFLINE" in os.environ:
        return
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    names = (settings.embed_model, settings.rerank_model)
    if all((hub / f"models--{n.replace('/', '--')}" / "snapshots").is_dir() for n in names):
        os.environ["HF_HUB_OFFLINE"] = "1"


_use_offline_hub_if_cached()

# These imports read HF_HUB_OFFLINE when they load, so they come after the check above.

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

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
