"""
Application configuration.

All settings can be overridden with environment variables or a .env file.
See .env.example for the variables that matter in practice.
"""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # Project paths
    # ---------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent

    business_docs_dir: Path = Field(
        default=Path("data/business_docs")
    )

    chroma_db_dir: Path = Field(
        default=Path("chroma_db")
    )

    # ---------------------------------------------------------
    # Vector store
    # ---------------------------------------------------------
    # The collection name and distance space are configured in ONE place.
    # ingest.py creates the collection with this space and rag_pipeline.py
    # asserts it on load, so the two can never silently disagree.
    collection_name: str = "smilecare_knowledge_base"
    distance_space: str = "cosine"

    # ---------------------------------------------------------
    # Embedding configuration
    # ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------------------------------------------------
    # Chunking configuration
    # ---------------------------------------------------------
    # 900/120 keeps the opening-hours markdown table intact. At the previous
    # 500/50 the table was split mid-row, which degraded answers about hours.
    chunk_size: int = 900
    chunk_overlap: int = 120

    # ---------------------------------------------------------
    # Retrieval configuration
    # ---------------------------------------------------------
    # With normalized embeddings and cosine space, similarity_search_with_score
    # returns a cosine DISTANCE in [0, 2]: 0 = identical, 1 = unrelated,
    # 2 = opposite. relevance = 1 - distance.
    #
    # retrieval_threshold is the maximum distance a chunk may have to be
    # considered relevant. Calibrate it with: python scripts/check_threshold.py
    top_k: int = 5
    retrieval_threshold: float = 0.70

    # Secondary filter: once the best match is known, drop chunks that are
    # much worse than it. Guards against padding good answers with weak chunks.
    relative_margin: float = 0.20

    # ---------------------------------------------------------
    # Reranking configuration
    # ---------------------------------------------------------
    enable_reranking: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 3

    # ---------------------------------------------------------
    # Conversation configuration
    # ---------------------------------------------------------
    max_history_messages: int = 15

    # ---------------------------------------------------------
    # LLM configuration
    # ---------------------------------------------------------
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
    )

    groq_model: str = "openai/gpt-oss-120b"

    temperature: float = 0.2
    max_tokens: int = 1024

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    log_level: str = "INFO"

    # ---------------------------------------------------------
    # Environment configuration
    # ---------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # Derived values
    # ---------------------------------------------------------
    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        """
        Resolve relative paths against base_dir.

        This runs during validation rather than as a module-level mutation,
        so paths are correct no matter how or where Settings is constructed
        and no matter what the current working directory is.
        """

        if not self.business_docs_dir.is_absolute():
            self.business_docs_dir = self.base_dir / self.business_docs_dir

        if not self.chroma_db_dir.is_absolute():
            self.chroma_db_dir = self.base_dir / self.chroma_db_dir

        return self

    @property
    def index_meta_path(self) -> Path:
        """Fingerprint file describing how the current index was built."""

        return self.chroma_db_dir / "index_meta.json"

    @property
    def has_groq_api_key(self) -> bool:
        return bool(self.groq_api_key.strip())


settings = Settings()
