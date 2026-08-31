"""
Retrieval threshold calibration.

Prints the cosine distance of the top matches for a set of questions that
SHOULD be answerable and a set that should NOT be, then recommends a
threshold that separates them.

Use this instead of guessing at config.retrieval_threshold:

    python scripts/check_threshold.py

Interpretation (cosine distance, lower is better):
    0.0  identical
    ~0.3 strongly related
    ~0.6 loosely related
    1.0  unrelated
    2.0  opposite

A healthy knowledge base shows a clear gap between the worst in-domain
distance and the best out-of-domain distance. Put the threshold in that gap.
If there is no gap, the fix is better documents or better chunking, not a
different number.
"""

import logging
import sys
from pathlib import Path

# Allow running this file directly from the repo root or from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings          # noqa: E402
from ingest import ensure_index      # noqa: E402
from rag_pipeline import load_vectorstore, reset_caches   # noqa: E402


IN_DOMAIN_QUESTIONS = [
    "What dental services does SmileCare provide?",
    "What are the clinic opening hours?",
    "How can I book an appointment?",
    "What insurance plans does SmileCare accept?",
    "Where is SmileCare Dental Clinic located?",
    "Do you treat children?",
    "How much does a dental cleaning cost?",
    "What should I do in a dental emergency?",
]

OUT_OF_DOMAIN_QUESTIONS = [
    "What is the weather in Lahore today?",
    "Who is the president of Pakistan?",
    "How do I cook biryani?",
    "What is the capital of France?",
    "Write me a Python script to sort a list.",
    "What time does the cricket match start?",
]


def top_distances(vectorstore, question):
    """Return (distance, source) pairs for a question, best first."""

    results = vectorstore.similarity_search_with_score(
        question,
        k=settings.top_k,
    )

    return [
        (float(distance), document.metadata.get("source", "unknown"))
        for document, distance in results
    ]


def report(vectorstore, title, questions):
    """Print per-question detail and return the best distance for each."""

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    best_distances = []

    for question in questions:

        matches = top_distances(vectorstore, question)

        print()
        print(question)

        if not matches:
            print("  (no candidates returned)")
            continue

        for distance, source in matches:
            print(f"  {distance:.4f}  {source}")

        best_distances.append(matches[0][0])

    return best_distances


def recommend(in_domain, out_of_domain):
    """Suggest a threshold from the observed separation."""

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if not in_domain or not out_of_domain:
        print("Not enough data to make a recommendation.")
        return

    worst_good = max(in_domain)
    best_bad = min(out_of_domain)

    print(f"Worst in-domain best-match distance : {worst_good:.4f}")
    print(f"Best out-of-domain best-match distance: {best_bad:.4f}")
    print(f"Configured retrieval_threshold        : "
          f"{settings.retrieval_threshold:.4f}")

    if best_bad <= worst_good:
        print()
        print(
            "No clean separation: at least one unrelated question scores "
            "better than a genuine one.\n"
            "No threshold can fix this. Improve the knowledge base coverage "
            "or the chunking instead."
        )
        return

    suggested = round((worst_good + best_bad) / 2, 2)

    print()
    print(f"Suggested retrieval_threshold: {suggested:.2f}")
    print(f"  (midpoint of the {best_bad - worst_good:.4f}-wide gap)")

    if not worst_good < settings.retrieval_threshold < best_bad:
        print()
        print(
            "The configured threshold falls OUTSIDE the usable gap. "
            "Too low rejects real questions; too high answers unrelated ones."
        )


def main():
    logging.basicConfig(
        level=settings.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if ensure_index():
        reset_caches()

    vectorstore = load_vectorstore()

    print()
    print(f"Collection    : {settings.collection_name}")
    print(f"Distance space: {settings.distance_space}")
    print(f"Embeddings    : {settings.embedding_model}")
    print(f"Chunking      : size={settings.chunk_size} "
          f"overlap={settings.chunk_overlap}")

    in_domain = report(
        vectorstore,
        "IN-DOMAIN (these must be answered)",
        IN_DOMAIN_QUESTIONS,
    )

    out_of_domain = report(
        vectorstore,
        "OUT-OF-DOMAIN (these must be refused)",
        OUT_OF_DOMAIN_QUESTIONS,
    )

    recommend(in_domain, out_of_domain)


if __name__ == "__main__":
    main()
