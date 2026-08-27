from config import settings
from rag_pipeline import retrieve_documents


def test_relevant_question_passes_threshold():
    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    assert len(documents) > 0

    for document in documents:
        assert (
            document.metadata["retrieval_score"]
            <= settings.retrieval_threshold
        )


def test_empty_question_returns_no_documents():
    documents = retrieve_documents("")

    assert documents == []


def test_whitespace_question_returns_no_documents():
    documents = retrieve_documents("   ")

    assert documents == []

def test_unrelated_question_fails_threshold():
    documents = retrieve_documents(
        "What is the weather in Lahore today?"
    )

    assert documents == []