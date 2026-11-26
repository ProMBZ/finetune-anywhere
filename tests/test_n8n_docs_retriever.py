from scripts.n8n_docs_retriever import augment_messages, rank_chunks


def test_rank_chunks_prefers_relevant_official_doc_excerpt():
    chunks = [
        {"source_id": "code", "url": "https://docs.n8n.io/code/code-node.md",
         "text": "The Code node reads input items with the supported input API and returns n8n item objects."},
        {"source_id": "merge", "url": "https://docs.n8n.io/build/flow-logic/merge-data.md",
         "text": "The Merge node combines data from multiple streams and can wait for its inputs."},
    ]
    results = rank_chunks("Code node input items and item objects", chunks, top_k=1)
    assert len(results) == 1
    assert results[0]["source_id"] == "code"


def test_augment_messages_adds_context_without_mutating_original():
    chunks = [
        {"source_id": "expressions", "url": "https://docs.n8n.io/code/expressions.md",
         "text": "Expressions reference data from previous nodes and use the expression editor syntax."},
    ]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": '{"workflow":{"nodes":[]},"execution_error":"expression is undefined","expected_behavior":"reference previous node"}'},
    ]
    augmented = augment_messages(messages, chunks, top_k=1, excerpt_chars=120)
    assert "documentation_context" in augmented[-1]["content"]
    assert "documentation_context" not in messages[-1]["content"]
