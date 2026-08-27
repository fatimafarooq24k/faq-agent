from config import settings
from rag_pipeline import retrieve_documents


# =========================================================
# Relevant questions
# =========================================================

def test_relevant_question_passes_threshold():
    questions = [
        "What dental services does SmileCare provide?",
        "What are the clinic opening hours?",
        "How can I book an appointment?",
        "What insurance plans does SmileCare accept?",
        "Where is SmileCare Dental Clinic located?",
    ]

    for question in questions:
        documents = retrieve_documents(question)

        assert len(documents) > 0, (
            f"No documents retrieved for relevant question: {question}"
        )

        for document in documents:
            assert (
                document.metadata["retrieval_score"]
                <= settings.retrieval_threshold
            )


# =========================================================
# Empty questions
# =========================================================

def test_empty_question_returns_no_documents():
    documents = retrieve_documents("")

    assert documents == []


def test_whitespace_question_returns_no_documents():
    documents = retrieve_documents("   ")

    assert documents == []


# =========================================================
# Out-of-domain questions
# =========================================================

def test_unrelated_question_fails_threshold():
    questions = [
        "What is the weather in Lahore today?",
        "Who is the president of Pakistan?",
        "How do I cook biryani?",
        "What is the capital of France?",
    ]

    for question in questions:
        documents = retrieve_documents(question)

        assert documents == [], (
            f"Unrelated question incorrectly retrieved documents: {question}"
        )


# =========================================================
# Metadata
# =========================================================

def test_retrieval_has_metadata():
    documents = retrieve_documents(
        "What are the clinic opening hours?"
    )

    assert len(documents) > 0

    for document in documents:
        assert "source" in document.metadata
        assert "document_type" in document.metadata
        assert "business" in document.metadata
        assert "retrieval_score" in document.metadata


# =========================================================
# Threshold validation
# =========================================================

def test_retrieval_scores_respect_threshold():
    documents = retrieve_documents(
        "What services does SmileCare provide?"
    )

    for document in documents:
        score = document.metadata["retrieval_score"]

        assert score <= settings.retrieval_threshold