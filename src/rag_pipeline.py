from src.llm_client import LLMClient
from src.models import RagAnswer, RetrievedChunk
from src.prompts import INSUFFICIENT_EVIDENCE_MESSAGE, build_rag_prompt
from src.vector_store import VectorStore


class RagPipeline:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = 5) -> RagAnswer:
        retrieved = self.vector_store.search(question, top_k=top_k)
        useful = [result for result in retrieved if result.score > 0]
        if not useful:
            return RagAnswer(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                evidence=[],
                confidence="Low",
                retrieved_chunks=[],
            )

        context_lines = [_context_line(index, result) for index, result in enumerate(useful, start=1)]
        prompt = build_rag_prompt(question, context_lines)
        answer_text = self.llm_client.generate(prompt).strip()
        evidence = [
            _evidence_line(index, result)
            for index, result in enumerate(useful, start=1)
        ]
        return RagAnswer(
            answer=answer_text,
            evidence=evidence,
            confidence=_confidence_from_results(useful),
            retrieved_chunks=useful,
        )


def _context_line(index: int, result: RetrievedChunk) -> str:
    metadata = result.chunk.metadata
    title = metadata.get("paper_title", "Unknown paper")
    page = metadata.get("page_number", "unknown page")
    chunk_id = metadata.get("chunk_id", "unknown chunk")
    source = metadata.get("source_url", "")
    return (
        f"[{index}] {title}, page {page}, chunk {chunk_id}, source {source}\n"
        f"{result.chunk.text}"
    )


def _evidence_line(index: int, result: RetrievedChunk) -> str:
    metadata = result.chunk.metadata
    title = metadata.get("paper_title", "Unknown paper")
    page = metadata.get("page_number", "unknown page")
    chunk_id = metadata.get("chunk_id", "unknown chunk")
    source = metadata.get("source_url", "")
    return f"[{index}] {title}, page {page}, chunk {chunk_id}, {source}".strip()


def _confidence_from_results(results: list[RetrievedChunk]) -> str:
    top_score = max(result.score for result in results)
    if top_score >= 0.65 and len(results) >= 2:
        return "High"
    if top_score >= 0.25:
        return "Medium"
    return "Low"
