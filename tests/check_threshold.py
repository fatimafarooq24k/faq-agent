from rag_pipeline import load_vectorstore


vectorstore = load_vectorstore()

questions = [
    "What dental services does SmileCare provide?",
    "What are the clinic opening hours?",
    "How can I book an appointment?",
    "What insurance plans does SmileCare accept?",
    "Where is SmileCare Dental Clinic located?",
    "What is the weather in Lahore today?",
    "Who is the president of Pakistan?",
    "How do I cook biryani?",
    "What is the capital of France?",
]


for question in questions:

    results = vectorstore.similarity_search_with_score(
        question,
        k=5,
    )

    print("\n" + "=" * 70)
    print(question)

    for document, score in results:
        print(
            f"{score:.4f} | "
            f"{document.metadata.get('source', 'Unknown')}"
        )