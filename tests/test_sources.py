from rag_pipeline import retrieve_documents, get_source_info, get_sources


def test_source_info_contains_required_fields():
    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    assert len(documents) > 0

    source_info = get_source_info(documents[0])

    assert "source" in source_info
    assert "document_type" in source_info
    assert "business" in source_info

    assert source_info["source"]
    assert source_info["document_type"]
    assert source_info["business"]


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