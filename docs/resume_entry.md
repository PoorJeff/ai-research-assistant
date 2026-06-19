# AI Research Assistant Resume Entry

## English Resume Bullet

Built a local-first AI Research Assistant for academic literature workflows, including arXiv search, one-click PDF ingestion, PyMuPDF text extraction, chunking, sentence-transformer embeddings, ChromaDB vector retrieval, and evidence-grounded RAG question answering with cited source chunks. Added Streamlit product UI, Docker packaging, pytest/ruff/GitHub Actions CI, and a reproducible real-paper retrieval evaluation over 3 arXiv papers and 53 parsed pages.

## Chinese Resume Bullet

构建本地优先的 AI Research Assistant，支持 arXiv 论文搜索、一键 PDF 导入、PyMuPDF 文本解析、分块、SentenceTransformer 向量化、ChromaDB 检索，以及带证据引用的 RAG 问答。完善 Streamlit 产品界面、Docker 部署、pytest/ruff/GitHub Actions CI，并基于 3 篇真实 arXiv 论文和 53 页解析文本完成可复现实验评估。

## Short Version

AI Research Assistant: local-first academic RAG app with arXiv search, one-click PDF ingestion, ChromaDB retrieval, citation-backed answers, Streamlit UI, Docker packaging, CI, and real-paper Recall@k evaluation.

## Interview Talking Points

- Product scope: turns literature search, PDF reading, retrieval, summarization, comparison, and Q&A into one local workflow.
- RAG architecture: page-aware PDF parsing, metadata-preserving chunks, swappable embedding backends, ChromaDB or in-memory vector stores, and evidence-bounded prompts.
- Reliability: deterministic mock LLM for offline demos, real local/API LLM support through provider adapters, pytest coverage, ruff linting, compile checks, and GitHub Actions CI.
- Evaluation: reproducible benchmark over three real arXiv PDFs with 53 parsed pages, 61 indexed chunks, 6 curated questions, Recall@3/5, and citation coverage.
- Product polish: Overview dashboard, runtime/index status, one-click import from arXiv search results, sample demo, Docker run path, README screenshots, and documented limitations.
