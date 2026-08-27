from pathlib import Path

from pydantic import Field
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
    # Embedding configuration
    # ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------------------------------------------------
    # Chunking configuration
    # ---------------------------------------------------------
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ---------------------------------------------------------
    # Retrieval configuration
    # ---------------------------------------------------------
    # NOTE: with normalized embeddings + cosine distance, scores
    # range 0 (identical) to 2 (opposite). Tune this empirically
    # by logging real scores for known-good questions.
    top_k: int = 5
    retrieval_threshold: float = 0.65

    # ---------------------------------------------------------
    # Reranking configuration
    # ---------------------------------------------------------
    enable_reranking: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 3

    # ---------------------------------------------------------
    # Conversation configuration
    # ---------------------------------------------------------
    conversation_turns: int = 6
    max_history_messages: int = 15

    # ---------------------------------------------------------
    # LLM configuration
    # ---------------------------------------------------------
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY"
    )

    groq_model: str = "openai/gpt-oss-120b"

    temperature: float = 0.2
    max_tokens: int = 1024

    # ---------------------------------------------------------
    # Environment configuration
    # ---------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Convert relative paths into absolute project paths
settings.business_docs_dir = (
    settings.base_dir / settings.business_docs_dir
)

settings.chroma_db_dir = (
    settings.base_dir / settings.chroma_db_dir
)