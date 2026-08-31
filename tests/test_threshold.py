"""
Threshold and filtering semantics.

This file used to be a near-exact copy of test_retrieval.py: the same four
test names asserting the same things, so a real retrieval regression showed
up as two identical failures and neither file was authoritative.

test_retrieval.py now owns the "does retrieval work" tests. This file owns
the "is the threshold configured coherently, and does the filter maths do
what it claims" tests - most of which need no vector store at all.
"""

import pytest

from config import settings
from rag_pipeline import retrieve_documents


# =========================================================
# Configuration sanity
# =========================================================

def test_distance_space_is_cosine():
    """
    The threshold is expressed as a cosine distance. If the collection is
    built with Chroma's default 'l2' space, scores are squared L2 - exactly
    twice the cosine distance on normalized vectors - and the threshold
    silently rejects everything. That was the deployment bug.
    """

    assert settings.distance_space == "cosine"


def test_threshold_is_in_valid_cosine_range():
    """Cosine distance is bounded by [0, 2]; a useful cutoff is well under 1."""

    assert 0.0 < settings.retrieval_threshold < 1.0


def test_relative_margin_is_smaller_than_threshold():
    """
    The relative margin only narrows the cutoff. A margin wider than the
    absolute threshold could never bind, making it dead configuration.
    """

    assert 0.0 < settings.relative_margin < settings.retrieval_threshold


def test_top_k_is_positive():
    assert settings.top_k >= 1


# =========================================================
# Filter behaviour
# =========================================================

@pytest.mark.retrieval
def test_no_returned_document_exceeds_the_threshold():

    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    assert len(documents) > 0

    for document in documents:
        assert (
            document.metadata["retrieval_score"]
            <= settings.retrieval_threshold
        )


@pytest.mark.retrieval
def test_relative_margin_is_applied():
    """
    Every returned chunk must be within relative_margin of the best one.
    This is the filter that stops a good answer being padded out with
    weakly-related chunks.
    """

    documents = retrieve_documents(
        "What are the clinic opening hours?"
    )

    assert len(documents) > 0

    scores = [
        document.metadata["retrieval_score"]
        for document in documents
    ]

    best = min(scores)

    for score in scores:
        assert score <= best + settings.relative_margin + 1e-9


@pytest.mark.retrieval
def test_relevance_is_the_complement_of_distance():

    documents = retrieve_documents(
        "How can I book an appointment?"
    )

    assert len(documents) > 0

    for document in documents:
        distance = document.metadata["retrieval_score"]
        relevance = document.metadata["relevance"]

        assert relevance == pytest.approx(1.0 - distance)
