from collections.abc import Iterable
import math
from pathlib import Path
from typing import Any, Protocol

from src.embeddings import EmbeddingModel
from src.models import RetrievedChunk, TextChunk


class VectorStore(Protocol):
    def add_chunks(self, chunks: Iterable[TextChunk]) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self._items: list[tuple[TextChunk, list[float]]] = []

    def add_chunks(self, chunks: Iterable[TextChunk]) -> None:
        chunk_list = list(chunks)
        embeddings = self.embedding_model.embed_texts([chunk.text for chunk in chunk_list])
        self._items.extend(zip(chunk_list, embeddings))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0 or not self._items:
            return []
        query_embedding = self.embedding_model.embed_query(query)
        ranked = sorted(
            (
                RetrievedChunk(chunk=chunk, score=_cosine_similarity(query_embedding, embedding))
                for chunk, embedding in self._items
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        return ranked[:top_k]


class ChromaVectorStore:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_path: str | Path = "vectorstore",
        collection_name: str = "papers",
    ):
        import chromadb

        self.embedding_model = embedding_model
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: Iterable[TextChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return

        ids = [chunk.chunk_id or f"chunk-{index}" for index, chunk in enumerate(chunk_list)]
        documents = [chunk.text for chunk in chunk_list]
        metadatas = [_clean_metadata(chunk.metadata) for chunk in chunk_list]
        embeddings = self.embedding_model.embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        query_embedding = self.embedding_model.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            retrieved.append(
                RetrievedChunk(
                    chunk=TextChunk(text=document, metadata=dict(metadata or {})),
                    score=score,
                )
            )
        return retrieved


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        elif value is not None:
            clean[key] = str(value)
    return clean
