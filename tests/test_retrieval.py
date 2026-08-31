"""
Retrieval behaviour against the real vector store.

Scope note: this file owns "does retrieval return the right documents".
Threshold arithmetic and configuration coherence live in test_threshold.py,
which previously duplicated four of these tests verbatim.
"""

import pytest

from rag_pipeline import retrieve_documents


# =========================================================
# Guard clauses (no vector store needed)
# =========================================================

def test_empty_question_returns_no_documents():
    documents = retrieve_documents("")

    assert documents == []


def test_whitespace_question_returns_no_documents():
    documents = retrieve_documents("   ")

    assert documents == []


# =========================================================
# Relevant questions
# =========================================================

RELEVANT_QUESTIONS = [
    "What dental services does SmileCare provide?",
    "What are the clinic opening hours?",
    "How can I book an appointment?",
    "What insurance plans does SmileCare accept?",
    "Where is SmileCare Dental Clinic located?",
]


@pytest.mark.retrieval
@pytest.mark.parametrize("question", RELEVANT_QUESTIONS)
def test_relevant_question_retrieves_documents(question):
    """
    Parametrized so a failure names the one question that broke, instead of
    aborting the whole loop on the first assertion as the previous version
    did.
    """

    documents = retrieve_documents(question)

    assert len(documents) > 0, (
        f"No documents retrieved for relevant question: {question}"
    )


# =========================================================
# Out-of-domain questions
# =========================================================

UNRELATED_QUESTIONS = [
    "What is the weather in Lahore today?",
    "Who is the president of Pakistan?",
    "How do I cook biryani?",
    "What is the capital of France?",
]


@pytest.mark.retrieval
@pytest.mark.parametrize("question", UNRELATED_QUESTIONS)
def test_unrelated_question_retrieves_nothing(question):
    documents = retrieve_documents(question)

    assert documents == [], (
        f"Unrelated question incorrectly retrieved documents: {question}"
    )


# =========================================================
# Metadata
# =========================================================

@pytest.mark.retrieval
def test_retrieved_documents_carry_expected_metadata():
    documents = retrieve_documents(
        "What are the clinic opening hours?"
    )

    assert len(documents) > 0

    for document in documents:
        assert "source" in document.metadata
        assert "document_type" in document.metadata
        assert "business" in document.metadata
        assert "retrieval_score" in document.metadata
        assert "relevance" in document.metadata


@pytest.mark.retrieval
def test_results_are_ordered_best_first():
    documents = retrieve_documents(
        "What insurance plans does SmileCare accept?"
    )

    assert len(documents) > 0

    scores = [
        document.metadata["retrieval_score"]
        for document in documents
    ]

    assert scores == sorted(scores)
