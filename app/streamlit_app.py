from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.arxiv_search import papers_to_dataframe, save_metadata_csv, search_arxiv
from src.chunking import chunk_pages
from src.config import Settings
from src.embeddings import HashEmbeddingModel, SentenceTransformerEmbeddingModel
from src.evaluation import answer_has_citations, compute_recall_at_k
from src.llm_client import create_llm_client
from src.models import TextChunk
from src.paper_compare import compare_papers
from src.pdf_loader import PdfLoadError, load_pdf_pages
from src.rag_pipeline import RagPipeline
from src.summarizer import summarize_paper
from src.vector_store import ChromaVectorStore, InMemoryVectorStore


st.set_page_config(page_title="AI Research Assistant", layout="wide")


def init_state() -> None:
    defaults = {
        "papers": [],
        "chunks": [],
        "vector_store": None,
        "last_answer": None,
        "summaries": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_embedding_model(backend: str, model_name: str):
    if backend == "Hash demo":
        return HashEmbeddingModel(dimension=384)
    return SentenceTransformerEmbeddingModel(model_name)


def build_vector_store(settings: Settings, embedding_backend: str, store_backend: str):
    embedder = get_embedding_model(embedding_backend, settings.embedding_model)
    if store_backend == "Chroma":
        return ChromaVectorStore(embedder, persist_path=settings.chroma_path)
    return InMemoryVectorStore(embedder)


def chunk_count_by_title(chunks: list[TextChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        title = str(chunk.metadata.get("paper_title", "Unknown paper"))
        counts[title] = counts.get(title, 0) + 1
    return counts


def chunks_for_title(chunks: list[TextChunk], title: str) -> list[TextChunk]:
    return [chunk for chunk in chunks if chunk.metadata.get("paper_title") == title]


def render_sidebar(settings: Settings) -> tuple[str, str]:
    st.sidebar.header("Runtime")
    st.sidebar.caption("Default settings are local-first and API-free.")
    embedding_backend = st.sidebar.selectbox(
        "Embedding backend",
        ["SentenceTransformer", "Hash demo"],
        index=0,
        help="SentenceTransformer gives semantic retrieval; Hash demo is fast for setup checks.",
    )
    store_backend = st.sidebar.selectbox(
        "Vector store",
        ["Chroma", "In-memory"],
        index=0,
        help="Chroma persists local indexes; in-memory is useful for temporary demos.",
    )
    st.sidebar.write(f"LLM provider: `{settings.llm_provider}`")
    st.sidebar.write(f"Embedding model: `{settings.embedding_model}`")
    st.sidebar.write(f"Chroma path: `{settings.chroma_path}`")
    return embedding_backend, store_backend


def render_search_page() -> None:
    st.subheader("Search Papers")
    query = st.text_input("Research topic", value="retrieval augmented generation")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        max_results = st.slider("Max results", 1, 20, 5)
    with col_b:
        sort_by = st.selectbox("Sort by", ["relevance", "submitted_date", "last_updated_date"])

    if st.button("Search arXiv", type="primary"):
        try:
            papers = search_arxiv(query, max_results=max_results, sort_by=sort_by)
            st.session_state.papers = papers
            output_path = save_metadata_csv(papers, "data/papers/metadata.csv")
            st.success(f"Found {len(papers)} papers. Metadata saved to {output_path}.")
        except Exception as exc:
            st.error(f"arXiv search failed: {exc}")

    if st.session_state.papers:
        frame = papers_to_dataframe(st.session_state.papers)
        st.dataframe(frame, use_container_width=True)
        st.download_button(
            "Download metadata CSV",
            data=frame.to_csv(index=False),
            file_name="metadata.csv",
            mime="text/csv",
        )


def render_upload_page(settings: Settings, embedding_backend: str, store_backend: str) -> None:
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more research PDFs", type=["pdf"], accept_multiple_files=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        chunk_size = st.number_input("Chunk size (words)", min_value=100, max_value=1200, value=700, step=50)
    with col_b:
        chunk_overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=300, value=120, step=20)

    if st.button("Extract and chunk PDFs", type="primary", disabled=not uploaded_files):
        new_chunks: list[TextChunk] = []
        for uploaded_file in uploaded_files or []:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
                handle.write(uploaded_file.getbuffer())
                temp_path = Path(handle.name)
            try:
                pages = load_pdf_pages(temp_path)
                paper_title = Path(uploaded_file.name).stem
                paper_id = paper_title.lower().replace(" ", "-")
                new_chunks.extend(
                    chunk_pages(
                        pages,
                        paper_title=paper_title,
                        paper_id=paper_id,
                        source_url=f"uploaded://{uploaded_file.name}",
                        chunk_size_words=int(chunk_size),
                        chunk_overlap_words=int(chunk_overlap),
                    )
                )
            except PdfLoadError as exc:
                st.error(str(exc))
            finally:
                temp_path.unlink(missing_ok=True)

        st.session_state.chunks.extend(new_chunks)
        st.success(f"Added {len(new_chunks)} chunks. Total chunks: {len(st.session_state.chunks)}.")

    if st.button("Build / rebuild vector index", disabled=not st.session_state.chunks):
        try:
            store = build_vector_store(settings, embedding_backend, store_backend)
            store.add_chunks(st.session_state.chunks)
            st.session_state.vector_store = store
            st.success(f"Indexed {len(st.session_state.chunks)} chunks with {store_backend}.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    counts = chunk_count_by_title(st.session_state.chunks)
    if counts:
        st.write("Indexed chunk candidates")
        st.dataframe(pd.DataFrame([{"paper_title": k, "chunks": v} for k, v in counts.items()]))
        with st.expander("Preview chunks"):
            for chunk in st.session_state.chunks[:8]:
                st.markdown(f"**{chunk.metadata.get('paper_title')}** - `{chunk.metadata.get('chunk_id')}`")
                st.write(chunk.text[:900])


def render_ask_page(settings: Settings) -> None:
    st.subheader("Ask Questions")
    question = st.text_input("Question", value="What problem does retrieval augmented generation solve?")
    top_k = st.slider("Retrieved chunks", 1, 10, 5)

    if st.button("Answer with evidence", type="primary"):
        if st.session_state.vector_store is None:
            st.warning("Build a vector index from uploaded PDFs first.")
            return
        llm_client = create_llm_client(settings)
        pipeline = RagPipeline(st.session_state.vector_store, llm_client)
        st.session_state.last_answer = pipeline.answer(question, top_k=top_k)

    if st.session_state.last_answer:
        answer = st.session_state.last_answer
        st.markdown(answer.to_markdown())
        with st.expander("Retrieved chunks"):
            for result in answer.retrieved_chunks:
                st.markdown(
                    f"**{result.chunk.metadata.get('paper_title', 'Unknown paper')}** "
                    f"`score={result.score:.3f}`"
                )
                st.write(result.chunk.text)


def render_compare_page(settings: Settings) -> None:
    st.subheader("Compare Papers")
    counts = chunk_count_by_title(st.session_state.chunks)
    selected_titles = st.multiselect("Papers from uploaded chunks", list(counts.keys()))
    manual = st.text_area(
        "Optional manual summaries, separated by ---",
        height=160,
        help="Use this when you want to compare summaries without calling the summarizer first.",
    )

    if st.button("Generate comparison", type="primary"):
        llm_client = create_llm_client(settings)
        summaries: list[str] = []
        for title in selected_titles:
            if title not in st.session_state.summaries:
                st.session_state.summaries[title] = summarize_paper(
                    title,
                    chunks_for_title(st.session_state.chunks, title),
                    llm_client,
                )
            summaries.append(f"Paper: {title}\n{st.session_state.summaries[title]}")

        summaries.extend(part.strip() for part in manual.split("---") if part.strip())
        st.markdown(compare_papers(summaries, llm_client))

    if st.session_state.summaries:
        with st.expander("Generated summaries"):
            for title, summary in st.session_state.summaries.items():
                st.markdown(f"### {title}")
                st.markdown(summary)


def render_evaluation_page() -> None:
    st.subheader("Evaluation")
    st.write("Use this page for lightweight checks after indexing a paper set.")
    st.markdown(
        """
1. Run the demo questions from `reports/demo_queries.md`.
2. Record expected paper titles or keywords.
3. Compare retrieved top-k items against expected items.
4. Check that answers include numbered evidence.
"""
    )

    retrieved = st.text_input("Retrieved items, comma-separated", value="paper-a, paper-b, paper-c")
    expected = st.text_input("Expected relevant items, comma-separated", value="paper-b, paper-x")
    k = st.slider("k", 1, 10, 3)
    if st.button("Compute Recall@k"):
        retrieved_items = [item.strip() for item in retrieved.split(",") if item.strip()]
        expected_items = {item.strip() for item in expected.split(",") if item.strip()}
        score = compute_recall_at_k(retrieved_items, expected_items, k)
        st.metric(f"Recall@{k}", f"{score:.2f}")

    answer_text = st.text_area("Paste an answer to check citation presence", height=120)
    if answer_text:
        st.write("Has citations:", answer_has_citations(answer_text))


def main() -> None:
    init_state()
    settings = Settings.from_env()
    embedding_backend, store_backend = render_sidebar(settings)

    st.title("AI Research Assistant")
    st.caption("Literature search, PDF ingestion, local vector retrieval, and evidence-based RAG Q&A.")

    tabs = st.tabs(["Search Papers", "Upload PDFs", "Ask Questions", "Compare Papers", "Evaluation"])
    with tabs[0]:
        render_search_page()
    with tabs[1]:
        render_upload_page(settings, embedding_backend, store_backend)
    with tabs[2]:
        render_ask_page(settings)
    with tabs[3]:
        render_compare_page(settings)
    with tabs[4]:
        render_evaluation_page()


if __name__ == "__main__":
    main()
