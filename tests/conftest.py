import os

import pytest

import ingest
import rag_pipeline


@pytest.fixture(scope="session", autouse=True)
def vector_store():
    """
    Guarantee a valid index exists before any test runs.

    ensure_index() is a no-op when the persisted index already matches the
    current documents and settings, so this is cheap on repeat runs and
    self-healing on a cold one.
    """

    rebuilt = ingest.ensure_index()

    if rebuilt:
        # The lru_cache handles in rag_pipeline may point at the collection
        # that was just deleted and recreated.
        rag_pipeline.reset_caches()

    return rag_pipeline.load_vectorstore()


def pytest_collection_modifyitems(config, items):
    """
    Skip integration tests when there is no API key.

    Without this, a fork PR (or a local checkout with no .env) reports
    failures for tests that simply cannot run in that environment.
    """

    if os.environ.get("GROQ_API_KEY", "").strip():
        return

    skip = pytest.mark.skip(reason="GROQ_API_KEY is not set")

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_groq(monkeypatch):
    """
    Replace Groq streaming with a deterministic local generator.

    Lets the full pipeline - retrieval, filtering, context building,
    message assembly, source extraction - be tested end to end without
    network access, an API key, or a bill.
    """

    def fake_stream(messages):
        yield "This is "
        yield "a test answer."

    monkeypatch.setattr(rag_pipeline, "stream_completion", fake_stream)

    return fake_stream
