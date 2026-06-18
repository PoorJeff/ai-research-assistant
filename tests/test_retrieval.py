from src.embeddings import HashEmbeddingModel
from src.models import TextChunk
from src.vector_store import InMemoryVectorStore


def test_in_memory_retrieval_returns_ranked_chunks():
    embedder = HashEmbeddingModel(dimension=16)
    store = InMemoryVectorStore(embedder)
    chunks = [
        TextChunk(
            text="graph retrieval answer generation",
            metadata={"chunk_id": "c1", "paper_title": "GraphRAG"},
        ),
        TextChunk(
            text="image classification with contrastive learning",
            metadata={"chunk_id": "c2", "paper_title": "Vision"},
        ),
    ]

    store.add_chunks(chunks)
    results = store.search("retrieval generation", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.metadata["chunk_id"] == "c1"
    assert results[0].score > 0
