"""Project settings. Values can be overridden with environment variables or .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Data
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    category: str = "Video_Games"

    # Selection rules (see ingest.select)
    min_ratings: int = 50
    max_products: int = 4000
    reviews_per_product: int = 20
    min_review_chars: int = 40

    # Chunking
    chunk_tokens: int = 300
    chunk_overlap_tokens: int = 40

    # Models
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"

    # Vector store
    weaviate_version: str = "1.40.2"
    weaviate_data_dir: Path = ROOT / "weaviate_data"

    # Generation
    generator: str = "ollama:qwen3.5:4b"
    judge_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None


settings = Settings()
