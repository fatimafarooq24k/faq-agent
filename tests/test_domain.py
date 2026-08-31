"""Out-of-domain detection."""

import pytest

from rag_pipeline import is_out_of_domain, retrieve_documents


def test_empty_retrieval_is_out_of_domain():
    assert is_out_of_domain([]) is True


@pytest.mark.retrieval
def test_relevant_question_is_in_domain():
    documents = retrieve_documents(
        "What dental services does SmileCare provide?"
    )

    assert len(documents) > 0
    assert is_out_of_domain(documents) is False


@pytest.mark.retrieval
def test_unrelated_question_is_out_of_domain():
    documents = retrieve_documents(
        "What is the weather in Lahore today?"
    )

    assert documents == []
    assert is_out_of_domain(documents) is True
