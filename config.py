"""
Application configuration.

All settings can be overridden with environment variables or a .env file.
When deployed on Streamlit Cloud, GROQ_API_KEY can also be loaded
from Streamlit Secrets.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_streamlit_secret() -> str:
    """
    Get GROQ_API_KEY from Streamlit Secrets.

    Returns an empty string when:
    - Streamlit is not installed
    - the app is running outside Streamlit
    - the secret does not exist
    """
    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # Project paths
    # ---------------------------------------------------------

    base_dir: Path = Path(__file__).resolve().parent

    business_docs_dir: Path = Path("data/business_docs")

    chroma_db_dir: Path = Path("chroma_db")

    # ---------------------------------------------------------
    # Vector store
    # ---------------------------------------------------------
    # The collection name and distance space are configured in
    # ONE place. ingest.py creates the collection with this
    # space and rag_pipeline.py asserts it on load.

    collection_name: str = "smilecare_knowledge_base"

    distance_space: str = "cosine"

    # ---------------------------------------------------------
    # Embedding configuration
    # ---------------------------------------------------------

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------------------------------
    # Chunking configuration
    # ---------------------------------------------------------

    # 900/120 keeps the opening-hours markdown table intact.

    chunk_size: int = 900

    chunk_overlap: int = 120

    # ---------------------------------------------------------
    # Retrieval configuration
    # ---------------------------------------------------------
    # With normalized embeddings and cosine space,
    # similarity_search_with_score returns a cosine DISTANCE:
    #
    # 0 = identical
    # 1 = unrelated
    # 2 = opposite
    #
    # relevance = 1 - distance

    retrieval_threshold: float = 0.70

    top_k: int = 5

    # Secondary filter: once the best match is known, drop
    # chunks that are much worse than it.

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

    groq_api_key: str = ""

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
        case_sensitive=False,
    )

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def __init__(self, **kwargs):
        """
        Load configuration in the following order:

        1. Explicit constructor values
        2. Environment variables
        3. .env file
        4. Streamlit Secrets for GROQ_API_KEY
        5. Default values
        """

        # Pydantic loads:
        # - environment variables
        # - .env
        # - default values

        super().__init__(**kwargs)

        # -----------------------------------------------------
        # Streamlit Cloud fallback for Groq API key
        # -----------------------------------------------------
        #
        # If GROQ_API_KEY was not found through the normal
        # environment/.env mechanism, try Streamlit Secrets.

        if not self.groq_api_key.strip():
            self.groq_api_key = get_streamlit_secret().strip()

        # -----------------------------------------------------
        # Resolve paths
        # -----------------------------------------------------

        self._resolve_paths()

    # ---------------------------------------------------------
    # Derived values
    # ---------------------------------------------------------

    def _resolve_paths(self) -> None:
        """
        Resolve relative paths against base_dir.

        This ensures paths work regardless of the directory
        from which the application is started.
        """

        if not self.business_docs_dir.is_absolute():
            self.business_docs_dir = (
                self.base_dir / self.business_docs_dir
            )

        if not self.chroma_db_dir.is_absolute():
            self.chroma_db_dir = (
                self.base_dir / self.chroma_db_dir
            )

    @property
    def index_meta_path(self) -> Path:
        """
        Fingerprint file describing how the current index
        was built.
        """
        return self.chroma_db_dir / "index_meta.json"

    @property
    def has_groq_api_key(self) -> bool:
        """
        Return True when a Groq API key is configured.
        """
        return bool(self.groq_api_key.strip())


# -------------------------------------------------------------
# Global settings instance
# -------------------------------------------------------------

settings = Settings()
