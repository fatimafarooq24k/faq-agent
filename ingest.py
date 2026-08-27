from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


# -----------------------------
# Load documents
# -----------------------------

def load_documents():
    """
    Load all Markdown documents from the business knowledge base.
    """

    loader = DirectoryLoader(
        str(settings.business_docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    return documents


# -----------------------------
# Add metadata
# -----------------------------

def add_metadata(documents):
    """
    Add useful metadata to every document.
    """

    for document in documents:
        source = Path(document.metadata["source"]).name

        document.metadata["source"] = source
        document.metadata["business"] = "SmileCare Dental Clinic"

        if source == "clinic_information.md":
            document.metadata["document_type"] = "clinic_information"

        elif source == "services.md":
            document.metadata["document_type"] = "services"

        elif source == "appointments.md":
            document.metadata["document_type"] = "appointments"

        elif source == "insurance.md":
            document.metadata["document_type"] = "insurance"

        elif source == "faq.md":
            document.metadata["document_type"] = "faq"

        else:
            document.metadata["document_type"] = "general"

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
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# -----------------------------
# Create embeddings
# -----------------------------

def create_embeddings():
    """
    Load the embedding model.

    normalize_embeddings=True is required so that Chroma's
    cosine distance (set below) produces meaningful, bounded
    scores (0 = identical, 2 = opposite). Without this, raw
    L2 distance on unnormalized vectors can exceed typical
    thresholds even for a perfect match.
    """

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding model loaded.")

    return embeddings


# -----------------------------
# Store in ChromaDB
# -----------------------------

def store_documents(chunks, embeddings):
    """
    Create the ChromaDB vector store using cosine distance.
    """

    print("Creating ChromaDB vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_db_dir),
        collection_name="smilecare_knowledge_base",
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Documents stored in ChromaDB.")

    return vectorstore


# -----------------------------
# Main ingestion pipeline
# -----------------------------

def main():

    print("\n==============================")
    print("SmileCare Knowledge Base")
    print("==============================\n")

    documents = load_documents()

    if not documents:
        raise ValueError(
            f"No Markdown documents found in {settings.business_docs_dir}"
        )

    documents = add_metadata(documents)

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    store_documents(chunks, embeddings)

    print("\n==============================")
    print("Ingestion completed successfully.")
    print("==============================\n")


if __name__ == "__main__":
    main()