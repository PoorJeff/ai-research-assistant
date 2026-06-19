from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.arxiv_search import papers_to_dataframe, save_metadata_csv, search_arxiv
from src.chunking import chunk_pages
from src.config import Settings
from src.demo_data import build_sample_demo_store
from src.embeddings import HashEmbeddingModel, SentenceTransformerEmbeddingModel
from src.evaluation import answer_has_citations, compute_recall_at_k
from src.llm_client import create_llm_client
from src.models import TextChunk
from src.paper_compare import compare_papers
from src.paper_ingestion import PaperIngestionError, ingest_papers_from_search
from src.pdf_loader import PdfLoadError, load_pdf_pages
from src.product_metrics import load_real_paper_evaluation, summarize_real_paper_evaluation
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
        "library_mode": "empty",
        "active_embedding_backend": None,
        "active_store_backend": None,
        "last_import_records": [],
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


def reset_workspace_state() -> None:
    st.session_state.papers = []
    st.session_state.chunks = []
    st.session_state.vector_store = None
    st.session_state.last_answer = None
    st.session_state.summaries = {}
    st.session_state.library_mode = "empty"
    st.session_state.active_embedding_backend = None
    st.session_state.active_store_backend = None
    st.session_state.last_import_records = []


def library_status_label() -> tuple[str, str]:
    if st.session_state.vector_store is not None:
        return "Index ready", "success"
    if st.session_state.chunks:
        return "Chunks pending index", "warning"
    return "No library loaded", "info"


def render_library_status() -> None:
    counts = chunk_count_by_title(st.session_state.chunks)
    status_text, status_level = library_status_label()
    if status_level == "success":
        st.success(status_text)
    elif status_level == "warning":
        st.warning(status_text)
    else:
        st.info(status_text)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Papers in workspace", len(counts))
    col_b.metric("Chunks prepared", len(st.session_state.chunks))
    col_c.metric("Search results", len(st.session_state.papers))
    col_d.metric("Vector index", "Ready" if st.session_state.vector_store else "Not built")

    if st.session_state.active_embedding_backend or st.session_state.active_store_backend:
        st.caption(
            "Active index: "
            f"{st.session_state.active_embedding_backend or 'unknown embedding'} + "
            f"{st.session_state.active_store_backend or 'unknown store'}"
        )


def render_real_evaluation_summary(show_tables: bool = False) -> None:
    report = load_real_paper_evaluation()
    if report is None:
        st.info("No real-paper evaluation report found.")
        return

    summary = summarize_real_paper_evaluation(report)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Real papers", summary.paper_count)
    metric_cols[1].metric("Parsed pages", summary.page_count)
    metric_cols[2].metric("Indexed chunks", summary.chunk_count)
    metric_cols[3].metric("Recall@3", f"{summary.average_recall_at_3:.2f}")
    metric_cols[4].metric(
        "Citations",
        f"{summary.citation_coverage_count}/{summary.question_count}",
    )

    st.caption(
        "Latest run: "
        f"{format_timestamp(report.generated_at)} | "
        f"{report.embedding_backend} | LLM: {report.llm_provider}"
    )

    if not show_tables:
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "pages": paper.pages,
                    "chunks": paper.chunks,
                }
                for paper in report.papers
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "question": result.question,
                    "expected": ", ".join(result.expected_paper_ids),
                    "retrieved_at_3": ", ".join(result.retrieved_paper_ids_at_3),
                    "recall_at_3": result.recall_at_3,
                    "recall_at_5": result.recall_at_5,
                    "citations": result.citation_present,
                    "confidence": result.confidence,
                }
                for result in report.query_results
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def format_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value


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
    if st.sidebar.button("Reset workspace"):
        reset_workspace_state()
        st.rerun()
    return embedding_backend, store_backend


def render_overview_page(settings: Settings, embedding_backend: str, store_backend: str) -> None:
    st.subheader("Overview")

    top_cols = st.columns([2, 1])
    with top_cols[0]:
        st.markdown("#### Current library")
        render_library_status()
    with top_cols[1]:
        st.markdown("#### Runtime")
        st.metric("LLM provider", settings.llm_provider)
        st.metric("Embedding backend", embedding_backend)
        st.metric("Vector store", store_backend)

    st.markdown("#### Real-paper benchmark")
    render_real_evaluation_summary()


def paper_option_label(paper) -> str:
    return f"{paper.paper_id} | {paper.title}"


def render_search_page(settings: Settings, embedding_backend: str, store_backend: str) -> None:
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

        with st.expander("Import PDFs from these results"):
            st.caption("Download selected arXiv PDFs, extract text, chunk them, and rebuild the index.")
            labels = [paper_option_label(paper) for paper in st.session_state.papers]
            label_to_paper = dict(zip(labels, st.session_state.papers, strict=True))
            selected_labels = st.multiselect(
                "Papers to import",
                labels,
                default=labels[: min(1, len(labels))],
            )
            col_a, col_b = st.columns([1, 1])
            with col_a:
                chunk_size = st.number_input(
                    "Import chunk size (words)",
                    min_value=100,
                    max_value=1200,
                    value=700,
                    step=50,
                )
            with col_b:
                chunk_overlap = st.number_input(
                    "Import chunk overlap (words)",
                    min_value=0,
                    max_value=300,
                    value=120,
                    step=20,
                )

            if st.button("Download, extract, and index selected papers", disabled=not selected_labels):
                selected_papers = [label_to_paper[label] for label in selected_labels]
                try:
                    with st.spinner("Downloading PDFs and rebuilding the vector index..."):
                        new_chunks, records = ingest_papers_from_search(
                            selected_papers,
                            chunk_size_words=int(chunk_size),
                            chunk_overlap_words=int(chunk_overlap),
                        )
                        combined_chunks = [*st.session_state.chunks, *new_chunks]
                        store = build_vector_store(settings, embedding_backend, store_backend)
                        store.add_chunks(combined_chunks)

                    st.session_state.chunks = combined_chunks
                    st.session_state.vector_store = store
                    st.session_state.last_answer = None
                    st.session_state.library_mode = "arxiv_import"
                    st.session_state.active_embedding_backend = embedding_backend
                    st.session_state.active_store_backend = store_backend
                    st.session_state.last_import_records = records
                    st.success(
                        f"Imported {len(records)} papers and indexed {len(combined_chunks)} total chunks."
                    )
                except (PaperIngestionError, PdfLoadError, ValueError) as exc:
                    st.error(f"Import failed: {exc}")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

            if st.session_state.last_import_records:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "paper_id": record.paper_id,
                                "title": record.title,
                                "pages": record.pages,
                                "chunks": record.chunks,
                                "pdf_path": record.pdf_path,
                            }
                            for record in st.session_state.last_import_records
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


def render_upload_page(settings: Settings, embedding_backend: str, store_backend: str) -> None:
    st.subheader("Upload PDFs")
    st.write("Load a tiny built-in sample when you want to test the RAG flow before uploading papers.")
    if st.button("Load sample RAG demo"):
        store, chunks = build_sample_demo_store()
        st.session_state.chunks = chunks
        st.session_state.vector_store = store
        st.session_state.library_mode = "sample_demo"
        st.session_state.active_embedding_backend = "Hash demo"
        st.session_state.active_store_backend = "In-memory"
        st.session_state.last_import_records = []
        st.success("Loaded sample chunks and built a fast in-memory demo index.")

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
        st.session_state.vector_store = None
        st.session_state.last_answer = None
        st.session_state.library_mode = "uploaded_chunks"
        st.session_state.active_embedding_backend = None
        st.session_state.active_store_backend = None
        st.session_state.last_import_records = []
        st.success(f"Added {len(new_chunks)} chunks. Total chunks: {len(st.session_state.chunks)}.")

    if st.button("Build / rebuild vector index", disabled=not st.session_state.chunks):
        try:
            store = build_vector_store(settings, embedding_backend, store_backend)
            store.add_chunks(st.session_state.chunks)
            st.session_state.vector_store = store
            st.session_state.library_mode = "indexed_pdfs"
            st.session_state.active_embedding_backend = embedding_backend
            st.session_state.active_store_backend = store_backend
            st.success(f"Indexed {len(st.session_state.chunks)} chunks with {store_backend}.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    render_library_status()

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
    status_text, status_level = library_status_label()
    if status_level == "success":
        st.success(status_text)
    elif status_level == "warning":
        st.warning(status_text)
    else:
        st.info(status_text)

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
    st.markdown("#### Latest real-paper run")
    render_real_evaluation_summary(show_tables=True)

    st.markdown("#### Manual retrieval check")
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

    tabs = st.tabs(["Overview", "Search Papers", "Upload PDFs", "Ask Questions", "Compare Papers", "Evaluation"])
    with tabs[0]:
        render_overview_page(settings, embedding_backend, store_backend)
    with tabs[1]:
        render_search_page(settings, embedding_backend, store_backend)
    with tabs[2]:
        render_upload_page(settings, embedding_backend, store_backend)
    with tabs[3]:
        render_ask_page(settings)
    with tabs[4]:
        render_compare_page(settings)
    with tabs[5]:
        render_evaluation_page()


if __name__ == "__main__":
    main()
