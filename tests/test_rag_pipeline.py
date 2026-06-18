from src.embeddings import HashEmbeddingModel
from src.llm_client import MockLLMClient
from src.models import TextChunk
from src.rag_pipeline import RagPipeline
from src.vector_store import InMemoryVectorStore


def test_rag_pipeline_includes_evidence_and_confidence():
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=16))
    store.add_chunks(
        [
            TextChunk(
                text="RAG retrieves relevant chunks before generating an answer.",
                metadata={
                    "chunk_id": "c1",
                    "paper_title": "RAG Paper",
                    "page_number": 2,
                    "source_url": "file://rag.pdf",
                },
            )
        ]
    )
    pipeline = RagPipeline(vector_store=store, llm_client=MockLLMClient())

    answer = pipeline.answer("How does RAG work?", top_k=1)

    assert "Answer:" in answer.to_markdown()
    assert "Evidence:" in answer.to_markdown()
    assert "RAG Paper" in answer.to_markdown()
    assert answer.confidence in {"High", "Medium", "Low"}


def test_rag_pipeline_handles_empty_retrieval():
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=16))
    pipeline = RagPipeline(vector_store=store, llm_client=MockLLMClient())

    answer = pipeline.answer("What does the paper conclude?", top_k=3)

    assert answer.confidence == "Low"
    assert "insufficient" in answer.answer.lower()
    assert answer.evidence == []
