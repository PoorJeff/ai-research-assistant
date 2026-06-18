from src.demo_data import build_sample_demo_store, sample_demo_chunks


def test_sample_demo_chunks_include_source_metadata():
    chunks = sample_demo_chunks()

    assert len(chunks) >= 3
    assert all(chunk.metadata["paper_title"] for chunk in chunks)
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert all(chunk.metadata["source_url"].startswith("sample://") for chunk in chunks)


def test_build_sample_demo_store_returns_searchable_chunks():
    store, chunks = build_sample_demo_store()

    results = store.search("retrieval augmented generation evidence", top_k=1)

    assert chunks
    assert results
    assert "RAG" in results[0].chunk.metadata["paper_title"]
