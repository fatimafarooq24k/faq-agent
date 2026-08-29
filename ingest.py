"""
Knowledge-base ingestion.

Builds the ChromaDB vector store from the markdown files in
data/business_docs, and records a fingerprint of how it was built so the
app can detect a stale or incompatible index instead of trusting that a
non-empty directory means a working index.

Run directly to rebuild:  python ingest.py
"""

import hashlib
import json
import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


# Filename stem -> document_type. Replaces the previous if/elif chain.
DOCUMENT_TYPES = {
    "clinic_information": "clinic_information",
    "services": "services",
    "appointments": "appointments",
    "insurance": "insurance",
    "faq": "faq",
}

# Prefer splitting on markdown structure before falling back to whitespace,
# so headings and tables are not cut in half.
MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    " ",
    "",
]


# -----------------------------
# Load documents
# -----------------------------

def load_documents():
    """
    Load all Markdown documents from the business knowledge base.

    Files are read directly with pathlib rather than through
    langchain_community's DirectoryLoader. That drops two heavy
    dependencies and makes the `source` metadata deterministic.
    """

    docs_dir = settings.business_docs_dir

    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {docs_dir}"
        )

    documents = []

    for path in sorted(docs_dir.rglob("*.md")):

        text = path.read_text(encoding="utf-8")

        if not text.strip():
            logger.warning("Skipping empty document: %s", path.name)
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "business": "SmileCare Dental Clinic",
                    "document_type": DOCUMENT_TYPES.get(
                        path.stem,
                        "general",
                    ),
                },
            )
        )

    logger.info("Loaded %d documents from %s", len(documents), docs_dir)

    return documents


# -----------------------------
# Split documents
# -----------------------------

def split_documents(documents):
    """
    Split documents into smaller chunks for retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=MARKDOWN_SEPARATORS,
    )

    chunks = splitter.split_documents(documents)

    logger.info("Created %d chunks", len(chunks))

    return chunks


# -----------------------------
# Create embeddings
# -----------------------------

def create_embeddings():
    """
    Load the embedding model.

    normalize_embeddings=True is required so that cosine distance produces
    bounded, comparable scores (0 = identical, 1 = unrelated, 2 = opposite).
    rag_pipeline.create_embeddings() must stay identical to this.
    """

    logger.info("Loading embedding model: %s", settings.embedding_model)

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


# -----------------------------
# Index fingerprint
# -----------------------------

def build_fingerprint(documents):
    """
    Describe how the index is being built.

    Any change to the embedding model, chunking, distance space, or the
    source documents changes this fingerprint, which is how ensure_index()
    knows the persisted store no longer matches the code and data.
    """

    doc_hashes = {
        document.metadata["source"]: hashlib.sha256(
            document.page_content.encode("utf-8")
        ).hexdigest()
        for document in documents
    }

    return {
        "embedding_model": settings.embedding_model,
        "collection_name": settings.collection_name,
        "distance_space": settings.distance_space,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "documents": dict(sorted(doc_hashes.items())),
    }


def read_fingerprint():
    """Return the fingerprint of the persisted index, or None."""

    path = settings.index_meta_path

    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read index fingerprint at %s", path)
        return None


def write_fingerprint(fingerprint):
    settings.index_meta_path.write_text(
        json.dumps(fingerprint, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# Store in ChromaDB
# -----------------------------

def get_chroma_client():
    return chromadb.PersistentClient(path=str(settings.chroma_db_dir))


def store_documents(chunks, embeddings):
    """
    Create the ChromaDB vector store using the configured distance space.

    The existing collection is deleted first. This matters: Chroma only
    applies collection_metadata when a collection is CREATED, so writing
    into a pre-existing collection would silently keep that collection's
    old distance space and quietly invalidate the retrieval threshold.
    """

    client = get_chroma_client()

    try:
        client.delete_collection(settings.collection_name)
        logger.info("Deleted existing collection to rebuild it cleanly")
    except Exception:
        # Collection did not exist yet, which is the normal first-run case.
        pass

    logger.info(
        "Creating collection %r with space=%s",
        settings.collection_name,
        settings.distance_space,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_db_dir),
        collection_name=settings.collection_name,
        collection_metadata={"hnsw:space": settings.distance_space},
    )

    logger.info("Stored %d chunks in ChromaDB", len(chunks))

    return vectorstore


# -----------------------------
# Index health
# -----------------------------

def index_status():
    """
    Report whether the persisted index is usable.

    Returns (ready, reason). This replaces the old "does the directory
    exist and is it non-empty" check, which happily accepted a directory
    containing an empty, stale, or wrong-metric collection - the failure
    mode that made every question return "no information available".
    """

    if not settings.chroma_db_dir.is_dir():
        return False, "index directory does not exist"

    try:
        collection = get_chroma_client().get_collection(
            settings.collection_name
        )
    except Exception:
        return False, f"collection {settings.collection_name!r} not found"

    count = collection.count()

    if count == 0:
        return False, "collection contains no documents"

    # Only flag an explicit mismatch. Some chromadb versions do not expose
    # the space through collection.metadata, and absence is not a problem.
    space = (collection.metadata or {}).get("hnsw:space")

    if space is not None and space != settings.distance_space:
        return False, (
            f"collection uses distance space {space!r}, "
            f"expected {settings.distance_space!r}"
        )

    persisted = read_fingerprint()

    if persisted is None:
        return False, "index fingerprint is missing"

    expected = build_fingerprint(load_documents())

    if persisted != expected:
        return False, "index does not match current documents or settings"

    return True, f"ready ({count} chunks)"


def ensure_index(force=False):
    """
    Build the index if it is missing, empty, stale, or incompatible.

    Returns True if the index was (re)built.
    """

    if not force:

        ready, reason = index_status()

        if ready:
            logger.info("Vector store %s", reason)
            return False

        logger.warning("Rebuilding vector store: %s", reason)

    build_index()

    return True


def build_index():
    """Rebuild the vector store from scratch."""

    documents = load_documents()

    if not documents:
        raise ValueError(
            f"No Markdown documents found in {settings.business_docs_dir}"
        )

    fingerprint = build_fingerprint(documents)

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    store_documents(chunks, embeddings)

    # Written last, so a fingerprint on disk always means a completed build.
    write_fingerprint(fingerprint)

    logger.info("Ingestion completed successfully")


# -----------------------------
# Entry point
# -----------------------------

def main():
    logging.basicConfig(
        level=settings.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Running this file directly always means "rebuild now".
    ensure_index(force=True)


if __name__ == "__main__":
    main()


