from src.embeddings import HashEmbeddingModel
from src.models import TextChunk
from src.vector_store import InMemoryVectorStore


def sample_demo_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text=(
                "Retrieval augmented generation (RAG) retrieves relevant passages from an "
                "external knowledge source before generating an answer. This reduces unsupported "
                "claims by grounding generation in retrieved evidence."
            ),
            metadata={
                "paper_title": "Sample RAG Systems Survey",
                "paper_id": "sample-rag-survey",
                "page_number": 2,
                "chunk_id": "sample-rag-survey-p2-c0",
                "source_url": "sample://rag-survey",
            },
        ),
        TextChunk(
            text=(
                "Evaluation of RAG systems often includes retrieval metrics such as Recall@k "
                "and answer-level checks for citation coverage, faithfulness, and whether the "
                "answer stays within the provided context."
            ),
            metadata={
                "paper_title": "Sample RAG Evaluation Notes",
                "paper_id": "sample-rag-eval",
                "page_number": 4,
                "chunk_id": "sample-rag-eval-p4-c0",
                "source_url": "sample://rag-evaluation",
            },
        ),
        TextChunk(
            text=(
                "Graph-based retrieval can connect entities, claims, and citations before "
                "answer generation. This can improve multi-hop questions but introduces extra "
                "pipeline complexity and graph construction errors."
            ),
            metadata={
                "paper_title": "Sample GraphRAG Method",
                "paper_id": "sample-graphrag",
                "page_number": 5,
                "chunk_id": "sample-graphrag-p5-c0",
                "source_url": "sample://graphrag-method",
            },
        ),
    ]


def build_sample_demo_store() -> tuple[InMemoryVectorStore, list[TextChunk]]:
    chunks = sample_demo_chunks()
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=384))
    store.add_chunks(chunks)
    return store, chunks
