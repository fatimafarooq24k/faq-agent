"""Source extraction and de-duplication."""

import pytest

from langchain_core.documents import Document

from rag_pipeline import get_source_info, get_sources, retrieve_documents


# =========================================================
# Pure functions (no vector store needed)
# =========================================================

def test_source_info_falls_back_when_metadata_is_missing():
    """
    get_source_info() must not raise on a document with no metadata -
    otherwise one badly-ingested chunk takes down the whole answer.
    """

    info = get_source_info(Document(page_content="text", metadata={}))

    assert info["source"] == "Unknown source"
    assert info["document_type"] == "general"
    assert info["business"] == "SmileCare Dental Clinic"


def test_get_sources_deduplicates_by_source_name():
    documents = [
        Document(page_content="a", metadata={"source": "services.md"}),
        Document(page_content="b", metadata={"source": "services.md"}),
        Document(page_content="c", metadata={"source": "faq.md"}),
    ]

    sources = get_sources(documents)

    assert [source["source"] for source in sources] == [
        "services.md",
        "faq.md",
    ]


def test_get_sources_of_nothing_is_empty():
    assert get_sources([]) == []


# =========================================================
# Against the real vector store
# =========================================================

@pytest.mark.retrieval
def test_source_info_contains_required_fields():
    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    assert len(documents) > 0

    source_info = get_source_info(documents[0])

    assert source_info["source"]
    assert source_info["document_type"]
    assert source_info["business"]


@pytest.mark.retrieval
def test_sources_are_unique():
    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    sources = get_sources(documents)

    source_names = [
        source["source"]
        for source in sources
    ]

    assert len(source_names) == len(set(source_names))


@pytest.mark.retrieval
def test_sources_have_expected_structure():
    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    sources = get_sources(documents)

    for source in sources:
        assert set(source.keys()) == {
            "source",
            "document_type",
            "business",
        }
